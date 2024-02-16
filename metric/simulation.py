import logging
import os
import tempfile
import time
from subprocess import Popen

import numpy as np
import pybullet as p
import skvideo.io as skvio

logger = logging.getLogger(__name__)


def write_obj(filename, verticies, faces):
    with open(filename, "w") as fp:
        for v in verticies:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for f in faces + 1:  # faces are 1-based, not 0-based in obj files
            fp.write("f %d %d %d\n" % (f[0], f[1], f[2]))

def take_picture(renderer, camera_pos, target_pos, width=256, height=256, conn_id=None):
    up_vector = [0, 0, 1]
    view_matrix = p.computeViewMatrix(
        camera_pos, target_pos, up_vector, physicsClientId=conn_id
    )
    proj_matrix = p.computeProjectionMatrixFOV(fov=90, aspect=1, nearVal=0.05, farVal=2, physicsClientId=conn_id)
    w, h, rgba, depth, mask = p.getCameraImage(
        width=width,
        height=height,
        projectionMatrix=proj_matrix,
        viewMatrix=view_matrix,
        renderer=renderer,
        physicsClientId=conn_id,
    )
    return rgba

class MetricSimulation:
    def __init__(self, opt, cfg):
        self.opt = opt
        self.cfg = cfg

    def simulation_displacement(
        self,
        hand_fidx,
        hand_verts,
        hand_faces,
        object_fidx,
        obj_verts,
        obj_faces,
    ):
        # setup
        if self.opt.use_gui:
            conn_id = p.connect(p.GUI)
        else:
            conn_id = p.connect(p.DIRECT)
        p.resetSimulation(physicsClientId=conn_id)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=conn_id)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=conn_id)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.opt.time_per_step, physicsClientId=conn_id
        )
        p.setGravity(0, 0, 9.8, physicsClientId=conn_id)
        target_pos = [0, -1, 0]
        camera_pos = [0, -1.5, 0]
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=0, cameraTargetPosition=target_pos)

        # opencv coordinate -> robotics coordinate
        hand_verts[:, [1,2]] = hand_verts[:, [2,1]]
        obj_verts[:, [1,2]] = obj_verts[:, [2,1]]
        hand_verts[:, 2] = -hand_verts[:, 2]
        obj_verts[:, 2] = -obj_verts[:, 2]

        # hand
        base_tmp_dir = "tmp/"
        os.makedirs(base_tmp_dir, exist_ok=True)
        hand_tmp_fname = tempfile.mktemp(suffix=".obj", dir=base_tmp_dir)
        write_obj(hand_tmp_fname, hand_verts, hand_faces)
        hand_indicies = hand_faces.flatten().tolist()
        hand_collision_id = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=hand_tmp_fname,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            indices=hand_indicies,
            physicsClientId=conn_id,
        )
        hand_visual_id = p.createVisualShape(
            p.GEOM_MESH,
            fileName=hand_tmp_fname,
            rgbaColor=[0, 0, 1, 1],
            specularColor=[0, 0, 1],
            physicsClientId=conn_id,
        )
        hand_body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=hand_collision_id,
            baseVisualShapeIndex=hand_visual_id,
            physicsClientId=conn_id,
        )
        p.changeDynamics(
            hand_body_id,
            -1,
            lateralFriction=self.opt.hand.friction,
            restitution=self.opt.hand.restitution,
            physicsClientId=conn_id,
        )

        # object
        obj_tmp_fname = tempfile.mktemp(suffix=".obj", dir=base_tmp_dir)
        os.makedirs(base_tmp_dir, exist_ok=True)
        initial_pos = np.mean(obj_verts, axis=0)
        write_obj(obj_tmp_fname, obj_verts, obj_faces)

        vhacd_result = vhacd(
            obj_tmp_fname, self.opt.vhacd.path, resolution=self.opt.vhacd.resolution
        )
        if vhacd_result:
            logger.debug(
                f"v-hacd for hand {hand_fidx}, object {object_fidx} successfully completed"
            )
        else:
            logger.error(f"v-hacd for hand {hand_fidx}, object {object_fidx} failed")
            raise ValueError

        obj_collision_id = p.createCollisionShape(
            p.GEOM_MESH, fileName=obj_tmp_fname, physicsClientId=conn_id
        )
        obj_visual_id = p.createVisualShape(
            p.GEOM_MESH,
            fileName=obj_tmp_fname,
            rgbaColor=[1, 0, 0, 1],
            specularColor=[1, 0, 0],
            physicsClientId=conn_id,
        )
        obj_body_id = p.createMultiBody(
            baseMass=self.opt.object.mass,
            basePosition=initial_pos,
            baseCollisionShapeIndex=obj_collision_id,
            baseVisualShapeIndex=obj_visual_id,
            physicsClientId=conn_id,
        )
        p.changeDynamics(
            obj_body_id,
            -1,
            lateralFriction=self.opt.object.friction,
            restitution=self.opt.object.restitution,
            physicsClientId=conn_id,
        )

        # run simulation
        if self.opt.save_video:
            images = []
            if self.opt.use_gui:
                renderer = p.ER_BULLET_HARDWARE_OPENGL
            else:
                renderer = p.ER_TINY_RENDERER
            save_video_dir = os.path.join(self.cfg.output_dir, "evaluations")
            os.makedirs(save_video_dir, exist_ok=True)
            save_video_path = os.path.join(
                save_video_dir,
                f"hand_{hand_fidx:08d}_object_{object_fidx}_simulation.gif",
            )

        for _ in range(self.opt.num_steps):
            p.stepSimulation(physicsClientId=conn_id)
            if self.opt.save_video:
                img = take_picture(renderer, camera_pos=camera_pos, target_pos=target_pos, conn_id=conn_id)
                images.append(img)
            time.sleep(self.opt.wait_time)

        if self.opt.save_video:
            skvio.vwrite(save_video_path, np.array(images).astype(np.uint8))
            logger.info(f"Saved simulation video to {save_video_path}")

        final_pos, final_ori = p.getBasePositionAndOrientation(
            obj_body_id, physicsClientId=conn_id
        )
        distance = np.linalg.norm(final_pos - initial_pos)

        os.remove(hand_tmp_fname)
        os.remove(obj_tmp_fname)
        p.disconnect(physicsClientId=conn_id)

        return distance


def vhacd(
    filename,
    vhacd_path,
    resolution=1000,
    concavity=0.001,
    planeDownsampling=4,
    convexhullDownsampling=4,
    alpha=0.05,
    beta=0.0,
    maxhulls=1024,
    pca=0,
    mode=0,
    maxNumVerticesPerCH=64,
    minVolumePerCH=0.0001,
):

    cmd_line = (
        '"{}" --input "{}" --resolution {} --concavity {:g} '
        "--planeDownsampling {} --convexhullDownsampling {} "
        "--alpha {:g} --beta {:g} --maxhulls {:g} --pca {:b} "
        "--mode {:b} --maxNumVerticesPerCH {} --minVolumePerCH {:g} "
        '--output "{}" --log "/dev/null"'.format(
            vhacd_path,
            filename,
            resolution,
            concavity,
            planeDownsampling,
            convexhullDownsampling,
            alpha,
            beta,
            maxhulls,
            pca,
            mode,
            maxNumVerticesPerCH,
            minVolumePerCH,
            filename,
        )
    )
    logger.info(cmd_line)

    devnull = open(os.devnull, "wb")
    vhacd_process = Popen(
        cmd_line,
        bufsize=-1,
        close_fds=True,
        shell=True,
        stdout=devnull,
        stderr=devnull,
    )
    return 0 == vhacd_process.wait()
