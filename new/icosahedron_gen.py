import torch

class IcosahedralGrid:
    def __init__(self, subdivisions=0, device=None, use_cuda=True):
        if device is None:
            device = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')
        else:
            device = torch.device(device)
        self.device = device
        self.subdivisions = subdivisions

        (
            self.vertices,
            self.edge_index,
            self.faces,
            self.pool_maps,
            self.up_maps,
        ) = self._build_grid()

    def _make_icosahedron(self):
        t = (1.0 + 5.0 ** 0.5) / 2.0
        verts = torch.tensor([
            [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
            [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
            [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
        ], dtype=torch.float32, device=self.device)

        faces = torch.tensor([
            [0, 11, 5],[0, 5, 1],[0, 1, 7],[0, 7, 10],[0, 10, 11],
            [1, 5, 9],[5, 11, 4],[11, 10, 2],[10, 7, 6],[7, 1, 8],
            [3, 9, 4],[3, 4, 2],[3, 2, 6],[3, 6, 8],[3, 8, 9],
            [4, 9, 5],[2, 4, 11],[6, 2, 10],[8, 6, 7],[9, 8, 1],
        ], dtype=torch.long, device=self.device)

        verts = verts / verts.norm(dim=1, keepdim=True)
        return verts, faces

    def _build_grid(self):
        verts, faces = self._make_icosahedron()
        verts_list = [v.clone() for v in verts]
        faces_list = [tuple(face.tolist()) for face in faces]

        pool_maps = []
        up_maps = []

        for _ in range(self.subdivisions):
            prev_n = len(verts_list)
            midpoint_cache = {}
            new_faces = []
            pool_map = {i: [i] for i in range(prev_n)}
            up_map = {}

            def get_midpoint(i, j):
                key = (i, j) if i < j else (j, i)
                if key in midpoint_cache:
                    return midpoint_cache[key]
                m = (verts_list[i] + verts_list[j]) * 0.5
                m = m / m.norm()
                idx = len(verts_list)
                verts_list.append(m)
                parent = min(i, j)
                up_map[idx] = parent
                pool_map[parent].append(idx)
                midpoint_cache[key] = idx
                return idx

            for (a, b, c) in faces_list:
                ab = get_midpoint(a, b)
                bc = get_midpoint(b, c)
                ca = get_midpoint(c, a)
                new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])

            faces_list = new_faces
            pool_maps.append(pool_map)

            new_n = len(verts_list)
            up_map_tensor = torch.empty((new_n,), dtype=torch.long, device=self.device)
            up_map_tensor[:prev_n] = torch.arange(prev_n, device=self.device)
            for fine_idx, coarse_idx in up_map.items():
                up_map_tensor[fine_idx] = coarse_idx
            up_maps.append(up_map_tensor)

        V = torch.stack(verts_list, dim=0).to(self.device)

        edge_set = set()
        for (a, b, c) in faces_list:
            for (i, j) in [(a, b), (b, c), (c, a)]:
                if i != j:
                    key = (i, j) if i < j else (j, i)
                    edge_set.add(key)

        src, dst = zip(*[(i, j) for (i, j) in sorted(edge_set)])
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long, device=self.device)

        return V.float(), edge_index, faces_list, pool_maps, up_maps
    
