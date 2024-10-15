import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Any, Union, List
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    AmbientLights,
    DirectionalLights,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    Materials,
    TexturesVertex,
)

from scipy.spatial.transform import Rotation


def build_perspective_camera(R, T, width, height, focal, device):
    # R: Rotation matrix of shape (N, 3, 3)
    # T: Translation matrix of shape (N, 3)
    focal_length = (
        (
            2 * focal / min(height, width),
            2 * focal / min(height, width),
        ),
    )
    return PerspectiveCameras(
        focal_length=focal_length,
        R=R,
        T=T,
        image_size=((height, width),),
        device=device,
    )


def build_mesh_renderer(cameras, width, height, focal, device):
    # Define the settings for rasterization and shading.
    raster_settings = RasterizationSettings(
        # image_size=(height, width),   # (H, W)
        image_size=height,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Define the material
    materials = Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=64,
        device=device
    )

    # Place a directional light in front of the object.
    lights = DirectionalLights(device=device, direction=((0, 2, 3),))
    # lights = AmbientLights(ambient_color=((1.0, 1.0, 1.0),), device=device)

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials
        )
    )

    return renderer


def render_mesh(vertices, triangles, intrinsics, extrinsic, width, height, focal=512.0, renderer=None, device=None):
    ''' Render the mesh under camera coordinates
    vertices: (V, 3) or (1, V, 3), vertices of mesh
    faces: (F, 3), faces of mesh
    focal: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''

    if device is None:
        device = torch.device('cuda')

    vertices = torch.from_numpy(vertices).squeeze(0).to(device)
    faces = torch.from_numpy(triangles.astype(np.int64)).to(device)

    # print(vertices.shape)     torch.Size([6890, 3])
    # print(faces.shape)        torch.Size([13776, 3])

    # upside down the mesh
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
    rot = torch.from_numpy(rot).to(device)

    vertices = torch.matmul(rot, vertices.T).T

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(vertices)[None]  # (B, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)
    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)

    R = torch.from_numpy(extrinsic[np.newaxis, :3, :3])
    T = torch.from_numpy(extrinsic[np.newaxis, :3, 3])

    # print(R.shape)  # [4, 4]
    # print(T.shape)  # [4, 4]

    # Do rendering
    color_batch = renderer(mesh)  # [1, 512, 512, 4]

    # To Image
    valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
    image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
    image_vis_batch = (image_vis_batch * 255).cpu().numpy()

    color = image_vis_batch[0]
    valid_mask = valid_mask_batch[0].cpu().numpy()
    input_img = np.zeros_like(color[:, :, :3])
    alpha = 1.0
    image_vis = alpha * color[:, :, :3] * valid_mask + (1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

    image_vis = image_vis.astype(np.uint8)

    image = Image.fromarray(image_vis, mode='RGB')
    return image


class PyTorch3DRenderer(object):
    def __init__(
        self,
        device: torch.device,
        extra_rot: bool = False,
    ):
        self.device = device

        # 设置相机
        R, T = look_at_view_transform(dist=2.0, elev=0.0, azim=0.0, device=self.device)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        # 设置光源
        lights = self.build_lights(light_type='ambient')
        shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)

        # 设置光栅化和着色器
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

        # 设置默认的渲染器
        self.default_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=shader,
        )

    def build_fov_perspective_cameras(
        self,
        camera_params: dict,
        extra_rot: bool = False,
    ) -> FoVPerspectiveCameras:
        
        extrinsic = camera_params['extrinsic']  # torch.Tensor, (N, 4, 4)

        if extra_rot:
            # Setting 1: 额外的旋转矩阵 (绕z轴逆时针旋转180度)
            # rotation_angle = torch.pi  # 180度对应pi弧度
            # rotation_matrix = torch.tensor([
            #     [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            #     [np.sin(rotation_angle),  np.cos(rotation_angle), 0],
            #     [0, 0, 1]
            # ], dtype=torch.float32, device=self.device)
            # # 更新相机的旋转矩阵
            # R = torch.matmul(R, rotation_matrix)
            # Setting 2: 翻转y轴
            extrinsic[:, 0, :] *= -1

        R = extrinsic[:, :3, :3]
        T = extrinsic[:, :3, 3]

        znear = camera_params['z_near']
        zfar = camera_params['z_far']
        aspect_ratio = camera_params['aspect_ratio']

        fov = camera_params['fov'].mean().item()

        return FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            znear=znear,
            zfar=zfar,
            fov=fov,
            aspect_ratio=aspect_ratio,
        )

    def build_lights(
        self,
        light_type: str = 'ambient',
        ambient_color: Optional[Tuple] = ((1.0, 1.0, 1.0),),
        light_location: Optional[Tuple] = None,
        **kwargs,
    ):
        if light_type == 'ambient':
            return AmbientLights(
                ambient_color=ambient_color,
                device=self.device,
            )
        elif light_type == 'directional':
            direction = ((0, 1, 0),) if light_location is None else light_location
            return DirectionalLights(
                ambient_color=ambient_color,
                direction=direction,
                device=self.device,
            )
        elif light_type == 'point':
            location = ((0, 1, 0),) if light_location is None else light_location
            return PointLights(
                ambient_color=ambient_color,
                location=location,
                device=self.device,
            )
        else:
            assert 0, f"Invalid light type: {light_type}"

    def build_materials(self):
        return Materials(
            ambient_color=[[0.3, 0.3, 0.3]],
            diffuse_color=[[0.5, 0.5, 0.5]],
            specular_color=[[1.0, 1.0, 1.0]],  # High specular color for metallic shine
            shininess=100.0,  # Higher shininess value for more reflective surface
            device=self.device,
        )

    def build_mesh_renderer(
        self,
        camera_params: dict,
        light_type: str = 'ambient',
        lights: Optional[PointLights] = None,
        materials: Optional[Materials] = None,
        **kwargs,
    ) -> MeshRenderer:
        # Build cameras
        cameras = self.build_fov_perspective_cameras(
            camera_params,
            extra_rot=kwargs.get('extra_rot', False),
        )

        # Build lights
        if lights is None:
            lights = self.build_lights(
                light_type=light_type,
                **kwargs,
            )

        # Build materials
        if materials is None:
            materials = self.build_materials()
        
        # Build shader
        shader = SoftPhongShader(
            device=self.device,
            cameras=cameras,
            lights=lights,
            materials=materials,
        )

        # Build rasterizer
        image_size = (camera_params['image_height'], camera_params['image_width'])
        raster_settings = RasterizationSettings(
            image_size=image_size,
            bin_size=0,
        )
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

        # Build renderer
        return MeshRenderer(
            rasterizer=rasterizer,
            shader=shader,
        )

    def render(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        camera_params: Optional[dict] = None,
        vertex_colors: Optional[torch.Tensor] = None,
        output_type: str = 'tensor',
        **kwargs,
    ) -> Union[torch.Tensor, np.ndarray, List[Image.Image]]:
        """
        Renders an image using the given vertices, faces, and vertex colors.

        Args:
            vertices (torch.Tensor): A tensor of shape (N, 3) representing the 3D coordinates of the vertices.
            faces (torch.Tensor): A tensor of shape (M, 3) representing the vertex indices of each triangular face.
            vertex_colors (Optional[torch.Tensor]): A tensor of shape (N, 3) representing the RGB colors of each vertex. 
                Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The first array is the rendered image 
            with shape (H, W, 3), where H and W are the height and width of the image, respectively. The second array 
            is the rendered image with alpha channel, with shape (H, W, 4).
        """
        # Create textures
        if vertex_colors is not None:
            textures = TexturesVertex(verts_features=[vertex_colors])
        else:
            textures = None

        # Create meshes
        mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)

        # Create renderer
        renderer = self.build_mesh_renderer(camera_params, **kwargs)

        # Render the image
        images = renderer(mesh)
        images[:, :, :, 3] = (images[:, :, :, 3] > 1e-6).float()
        if output_type == 'tensor':
            return images
        
        # Convert the images to numpy arrays
        images = (255 * images.cpu().numpy()).astype(np.uint8)
        if output_type == 'ndarray':
            return images
        
        # Convert the images to PIL images
        if output_type == 'pil_image':
            images = [Image.fromarray(image) for image in images]
            return images

        assert 0, f"Invalid output type: {output_type}"
