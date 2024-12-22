from dataclasses import dataclass
import pygame
from OpenGL.GL import *
import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Set
import os


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
            # load the texture referred to by this declaration
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
    ):
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


def load_simplified_obj(
    filename: str, voxel_size: float, swapyz: bool = False, contraction: str = "average"
) -> OBJ:
    """Helper function to load and simplify an OBJ file in one step."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    if contraction not in ["average", "quadric"]:
        raise ValueError(f"Invalid contraction method: {contraction}")

    start_time = pygame.time.get_ticks()
    obj = OBJ(filename, swapyz=swapyz)
    original_vertices = len(obj.vertices)
    new_vertices = obj.simplify(voxel_size, SimplificationContraction(contraction))
    end_time = pygame.time.get_ticks()
    print(
        f"Simplified mesh from {original_vertices} to {new_vertices} vertices in {end_time - start_time} ms"
    )
    return obj
