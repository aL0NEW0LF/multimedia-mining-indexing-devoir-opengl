from dataclasses import dataclass
import pygame
from OpenGL.GL import *
import numpy as np
from enum import Enum
import os
from typing import List, Set, Optional, Dict, Tuple
from collections import defaultdict


class SimplificationContraction(Enum):
    Average = "average"
    Quadric = "quadric"


@dataclass
class Quadric:
    A: np.ndarray  # 3x3 matrix
    b: np.ndarray  # 3x1 vector
    c: float

    def __init__(self, plane: np.ndarray = None, area: float = 1.0):
        if plane is None:
            self.A = np.zeros((3, 3))
            self.b = np.zeros(3)
            self.c = 0.0
        else:
            n = plane[:3]
            d = plane[3]
            self.A = area * np.outer(n, n)
            self.b = area * d * n
            self.c = area * d * d

    def __add__(self, other):
        result = Quadric()
        result.A = self.A + other.A
        result.b = self.b + other.b
        result.c = self.c + other.c
        return result

    def is_invertible(self) -> bool:
        return np.linalg.matrix_rank(self.A) == 3

    def minimum(self) -> np.ndarray:
        try:
            return np.linalg.solve(self.A, -self.b)
        except np.linalg.LinAlgError:
            return None


@dataclass(eq=False)
class CollapseVertex:
    position: np.ndarray
    id: int
    neighbors: Set["CollapseVertex"]
    faces: Set["CollapseTriangle"]
    obj_dist: float = float("inf")
    collapse_to: Optional["CollapseVertex"] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, CollapseVertex):
            return self.id == other.id
        return False

    def remove_if_non_neighbor(self, n: "CollapseVertex"):
        if n not in self.neighbors:
            return
        for face in self.faces:
            if face.has_vertex(n):
                return
        self.neighbors.remove(n)


class CollapseTriangle:
    def __init__(self, v0: CollapseVertex, v1: CollapseVertex, v2: CollapseVertex):
        self.vertices = [v0, v1, v2]
        self.normal = self.compute_normal()

        for i in range(3):
            self.vertices[i].faces.add(self)
            for j in range(3):
                if i != j:
                    self.vertices[i].neighbors.add(self.vertices[j])

    def compute_normal(self) -> np.ndarray:
        v0, v1, v2 = [v.position for v in self.vertices]
        normal = np.cross(v1 - v0, v2 - v1)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 0 else np.zeros(3)

    def has_vertex(self, v: CollapseVertex) -> bool:
        return any(v.id == vertex.id for vertex in self.vertices)

    def replace_vertex(self, vold: CollapseVertex, vnew: CollapseVertex):
        assert vold in self.vertices and vnew not in self.vertices
        idx = self.vertices.index(vold)
        self.vertices[idx] = vnew

        vold.faces.remove(self)
        vnew.faces.add(self)

        for v in self.vertices:
            vold.remove_if_non_neighbor(v)
            v.remove_if_non_neighbor(vold)

        for i in range(3):
            for j in range(3):
                if i != j:

                    self.vertices[i].neighbors.add(self.vertices[j])

        self.normal = self.compute_normal()


def compute_edge_collapse_cost(u: CollapseVertex, v: CollapseVertex) -> float:
    edge_length = np.linalg.norm(v.position - u.position)
    curvature = 0.0

    sides = [face for face in u.faces if face.has_vertex(v)]

    for face in u.faces:
        min_curv = 1.0
        for side in sides:
            dot_prod = np.dot(face.normal, side.normal)
            min_curv = min(min_curv, (1.0 - dot_prod) / 2.0)
        curvature = max(curvature, min_curv)

    return edge_length * curvature


def compute_vertex_collapse_cost(v: CollapseVertex):
    if not v.neighbors:
        v.collapse_to = None
        v.obj_dist = -0.01
        return

    v.obj_dist = float("inf")
    v.collapse_to = None

    for neighbor in v.neighbors:
        dist = compute_edge_collapse_cost(v, neighbor)
        if dist < v.obj_dist:
            v.collapse_to = neighbor
            v.obj_dist = dist


def find_minimum_cost_edge(vertices: List[CollapseVertex]) -> Optional[CollapseVertex]:
    if not vertices:
        return None
    return min(vertices, key=lambda v: v.obj_dist)


def collapse_edge(
    u: CollapseVertex, vertices: List[CollapseVertex], triangles: List[CollapseTriangle]
):
    if not u.collapse_to:

        vertices.remove(u)
        return

    v = u.collapse_to
    neighbors = list(u.neighbors)

    faces_to_remove = [face for face in u.faces if face.has_vertex(v)]
    for face in faces_to_remove:
        if face in triangles:
            triangles.remove(face)

    remaining_faces = [face for face in u.faces if face not in faces_to_remove]
    for face in remaining_faces:
        face.replace_vertex(u, v)

    vertices.remove(u)

    for neighbor in neighbors:
        compute_vertex_collapse_cost(neighbor)


def MTL(filename):
    contents = {}
    mtl = None
    for line in open(filename, "r"):
        if line.startswith("#"):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == "newmtl":
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError("mtl file doesn't start with newmtl stmt")
        elif values[0] == "map_Kd":
            mtl[values[0]] = values[1]
            surf = pygame.image.load(mtl["map_Kd"])
            image = pygame.image.tostring(surf, "RGBA", 1)
            ix, iy = surf.get_rect().size
            texid = mtl["texture_Kd"] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image
            )
        else:
            mtl[values[0]] = map(float, values[1:])
    return contents


"""
class OBJ:
    def __init__(self, filename, swapyz=False):
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.mtl = {}

        material = None
        for line in open(filename, "r"):
            if line.startswith("#"):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == "v":
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == "vn":
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == "vt":
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ("usemtl", "usemat"):
                material = values[1]
            elif values[0] == "mtllib":
                self.mtl = MTL(values[1])
            elif values[0] == "f":
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split("/")
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        for face in self.faces:
            vertices, normals, texture_coords, material = face

            if material in self.mtl:
                mtl = self.mtl[material]
                if "texture_Kd" in mtl:
                    # use diffuse texmap
                    glBindTexture(GL_TEXTURE_2D, mtl["texture_Kd"])
                else:
                    # just use diffuse colour
                    glColor(*mtl["Kd"])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()
        """


class OBJ:
    def __init__(self, filename, swapyz=False):
        self.vertices: List[List[float]] = []
        self.normals: List[List[float]] = []
        self.texcoords: List[List[float]] = []
        self.faces: List[Tuple] = []
        self.mtl = {}
        self.gl_list = 0
        self.swapyz = swapyz

        self.load_obj(filename)
        self.create_gl_list()

    def load_obj(self, filename):
        material = None
        for line in open(filename, "r"):
            if line.startswith("#"):
                continue
            values = line.split()
            if not values:
                continue

            if values[0] == "v":
                v = list(map(float, values[1:4]))
                if self.swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == "vn":
                v = list(map(float, values[1:4]))
                if self.swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == "vt":
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ("usemtl", "usemat"):
                material = values[1]
            elif values[0] == "mtllib":
                self.mtl = MTL(values[1])
            elif values[0] == "f":
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split("/")
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

    def get_triangle_plane(self, tidx: int) -> np.ndarray:
        triangle = self.faces[tidx]

        v0 = np.array(self.vertices[triangle[0][0] - 1])
        v1 = np.array(self.vertices[triangle[0][1] - 1])
        v2 = np.array(self.vertices[triangle[0][2] - 1])

        normal = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return np.array([0, 0, 0, 0])
        normal = normal / norm
        d = -np.dot(normal, v0)
        return np.array([*normal, d])

    def get_triangle_area(self, tidx: int) -> float:
        triangle = self.faces[tidx]

        v0 = np.array(self.vertices[triangle[0][0] - 1])
        v1 = np.array(self.vertices[triangle[0][1] - 1])
        v2 = np.array(self.vertices[triangle[0][2] - 1])

        return 0.5 * np.linalg.norm(np.cross(v0 - v1, v0 - v2))

    def create_gl_list(self):
        """Creates an OpenGL display list for the object"""
        if self.gl_list != 0:
            glDeleteLists(self.gl_list, 1)

        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)

        for face in self.faces:
            vertices, normals, texture_coords, material = face

            if material in self.mtl:
                mtl = self.mtl[material]
                if "texture_Kd" in mtl:
                    glBindTexture(GL_TEXTURE_2D, mtl["texture_Kd"])
                else:
                    glColor(*mtl["Kd"])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()

        glDisable(GL_TEXTURE_2D)
        glEndList()

    def simplify(
        self,
        voxel_size: float,
        contraction: SimplificationContraction = SimplificationContraction.Average,
    ) -> int:
        """Simplifies the mesh using vertex clustering."""
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")

        vertices = np.array(self.vertices)

        voxel_min = vertices.min(axis=0) - voxel_size * 0.5
        voxel_max = vertices.max(axis=0) + voxel_size * 0.5

        def get_voxel_idx(vert):
            ref_coord = (vert - voxel_min) / voxel_size
            return tuple(map(int, np.floor(ref_coord)))

        voxel_vertices: Dict[Tuple[int, int, int], Set[int]] = {}
        voxel_vert_ind: Dict[Tuple[int, int, int], int] = {}
        new_vertices = []
        vertex_map = {}

        for idx, vertex in enumerate(vertices):
            vox_idx = get_voxel_idx(vertex)
            if vox_idx not in voxel_vertices:
                voxel_vertices[vox_idx] = set()
                voxel_vert_ind[vox_idx] = len(new_vertices)
                if contraction == SimplificationContraction.Average:
                    new_vertices.append(vertex)
            voxel_vertices[vox_idx].add(idx)
            vertex_map[idx] = voxel_vert_ind[vox_idx]

        if contraction == SimplificationContraction.Average:
            for vox_idx, vert_indices in voxel_vertices.items():
                new_pos = np.mean([vertices[i] for i in vert_indices], axis=0)
                new_vertices[voxel_vert_ind[vox_idx]] = new_pos.tolist()

        elif contraction == SimplificationContraction.Quadric:
            vert_to_faces: Dict[int, Set[int]] = {}
            for face_idx, face in enumerate(self.faces):
                for vidx in face[0]:
                    if vidx not in vert_to_faces:
                        vert_to_faces[vidx] = set()
                    vert_to_faces[vidx].add(face_idx)

            for voxel_idx, vertex_indices in voxel_vertices.items():
                q = Quadric()
                for vidx in vertex_indices:
                    for tidx in vert_to_faces.get(vidx, []):
                        p = self.get_triangle_plane(tidx)
                        area = self.get_triangle_area(tidx)
                        q += Quadric(p, area)

                if q.is_invertible():
                    v = q.minimum()
                    if v is not None:
                        new_pos = v
                    else:
                        new_pos = np.mean([vertices[i] for i in vertex_indices], axis=0)
                else:
                    new_pos = np.mean([vertices[i] for i in vertex_indices], axis=0)

                new_vertices.append(new_pos.tolist())

        new_faces = []
        for face in self.faces:
            old_vertices, normals, texcoords, material = face
            new_face_vertices = [vertex_map[v - 1] + 1 for v in old_vertices]

            if len(set(new_face_vertices)) < 3:
                continue

            new_faces.append((new_face_vertices, normals, texcoords, material))

        self.vertices = new_vertices
        self.faces = new_faces

        self.create_gl_list()

        return len(self.vertices)

    def simplify_edge_collapse(self, target_vertices: int) -> int:
        """Simplifies the mesh using edge collapse until reaching target vertex count."""
        if target_vertices >= len(self.vertices):
            return len(self.vertices)

        collapse_vertices = []
        vertex_map = {}

        for i, vertex in enumerate(self.vertices):
            cv = CollapseVertex(
                position=np.array(vertex), id=i, neighbors=set(), faces=set()
            )
            collapse_vertices.append(cv)
            vertex_map[i] = cv

        collapse_triangles = []
        for face in self.faces:
            vertices = face[0]
            if len(vertices) != 3:
                continue

            triangle = CollapseTriangle(
                vertex_map[vertices[0] - 1],
                vertex_map[vertices[1] - 1],
                vertex_map[vertices[2] - 1],
            )
            collapse_triangles.append(triangle)

        for vertex in collapse_vertices:
            compute_vertex_collapse_cost(vertex)

        while len(collapse_vertices) > target_vertices:
            vertex = find_minimum_cost_edge(collapse_vertices)
            if not vertex:
                break
            collapse_edge(vertex, collapse_vertices, collapse_triangles)

        new_vertices = []
        new_vertex_map = {}

        for i, vertex in enumerate(collapse_vertices):
            new_vertices.append(vertex.position.tolist())
            new_vertex_map[vertex.id] = i

        print(f"Collapsed to {len(new_vertices)} vertices")

        new_faces = []
        for triangle in collapse_triangles:
            vertices = [new_vertex_map[v.id] + 1 for v in triangle.vertices]
            new_faces.append((vertices, [], [], None))

        self.vertices = new_vertices
        self.faces = new_faces
        self.create_gl_list()

        return len(self.vertices)


def load_simplified_obj(
    filename: str,
    voxel_size: float = 0.0,
    target_vertices: int = 0,
    swapyz: bool = False,
    contraction: str = "average",
    method: str = "vertex_clustering",
) -> OBJ:
    """Helper function to load and simplify an OBJ file in one step."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    if method not in ["vertex_clustering", "edge_collapse"]:
        raise ValueError(f"Invalid simplification method: {method}")

    if method != "vertex_clustering" and contraction not in ["average", "quadric"]:
        raise ValueError(f"Invalid contraction method: {contraction}")

    start_time = pygame.time.get_ticks()
    obj = OBJ(filename, swapyz=swapyz)
    original_vertices = len(obj.vertices)
    if method == "vertex_clustering":
        new_vertices = obj.simplify(voxel_size, SimplificationContraction(contraction))
    else:
        new_vertices = obj.simplify_edge_collapse(target_vertices)
    end_time = pygame.time.get_ticks()
    print(
        f"Simplified mesh from {original_vertices} to {new_vertices} vertices in {end_time - start_time} ms"
    )
    return obj
