import os
import numpy as np
from collections import namedtuple
from minieigen import Vector3, Matrix3, Quaternion


Pose = namedtuple('Pose', ['R', 't'])


def output_pose(datadir, pr_poses, gt_poses):
    poses = []
    for (pr, gt) in zip(pr_poses, gt_poses):
        t, R = pose_fusion([pr, gt])
        poses.append(Pose(R=R, t=t))

    pose_txt = datadir + 'avraged_pose.txt'
    with open(pose_txt, "w") as file:
        file.write("# averaged predicted poses\n")
        file.write("# file: 'freiburg desk dataset\n")
        file.write("# tx ty tz qx qy qz qw\n")
        for pose in poses:
            aa = rotation_matrix_to_angleaxis(pose.R)
            quat = angleaxis_to_quaternion(Vector3(aa))
            file.write("{} {} {} {} {} {} {}\n".format(pose.t[0],
                       pose.t[1], pose.t[2], quat[0], quat[1], quat[2], quat[3]))


def rotation_matrix_to_angleaxis(R):
    """Converts the rotation matrix to an angle axis vector with the angle
    encoded as the magnitude.

    R: minieigen.Matrix3 or np.array

    returns an angle axis vector as np.array
    """
    angle, axis = Quaternion(R).toAngleAxis()
    aa = angle * axis
    return np.array(aa)


def angleaxis_to_angle_axis(aa, epsilon=1e-6):
    """Converts the angle axis vector with angle encoded as magnitude to
    the angle axis representation with seperate angle and axis.

    aa: minieigen.Vector3
        axis angle with angle as vector magnitude

    epsilon: minimum angle in rad
        If the angle is smaller than epsilon
        then 0,(1,0,0) will be returned

    returns the tuple (angle,axis)
    """
    angle = aa.norm()
    if angle < epsilon:
        angle = 0
        axis = Vector3(1, 0, 0)
    else:
        axis = aa.normalized()
    return angle, axis


def angleaxis_to_quaternion(aa, epsilon=1e-6):
    """Converts the angle axis vector with angle encoded as magnitude to
    the quaternion representation.

    aa: minieigen.Vector3
        axis angle with angle as vector magnitude

    epsilon: minimum angle in rad
        If the angle is smaller than epsilon
        then 0,(1,0,0) will be returned

    returns the unit quaternion
    """
    angle, axis = angleaxis_to_angle_axis(aa, epsilon)
    return Quaternion(angle, axis)


def pose_fusion(pr_poses):
    '''
    Average the predicted poses of each camera
    pr_poses: list of Pose
    '''
    poses_t = np.zeros((len(pr_poses), 3))
    poses_aa = np.zeros((len(pr_poses), 3))
    cam_num = 0

    for pose in pr_poses:
        poses_aa[cam_num] = rotation_matrix_to_angleaxis(pr_poses[cam_num].R)
        poses_t[cam_num] = pr_poses[cam_num].t
        cam_num = cam_num + 1

    aa = Vector3(np.mean(poses_aa, axis=0))
    t = Vector3(np.mean(poses_t, axis=0))

    return t, angleaxis_to_quaternion(aa)


def main():
    R1 = Matrix3(1, 0, 0, 0, 0.7071089, -0.7071047, 0, 0.7071047, 0.7071089)
    R2 = Matrix3(0, -1, 0, 1, 0, 0, 0, 0, 1)
    R3 = Matrix3(0, 1, 0, -1, 0, 0, 0, 0, 1)
    R4 = Matrix3(0, 0, -1, 0, 1, 0, 1, 0, 0)
    R5 = Matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1)
    R11 = Matrix3(1, 0, 0, 0, 0.7071089, 0.7071047, 0, -0.7071047, 0.7071089)
    R22 = Matrix3(0, 1, 0, -1, 0, 0, 0, 0, -1)
    R33 = Matrix3(0, -1, 0, 1, 0, 0, 0, 0, -1)
    R44 = Matrix3(0, 0, 1, 0, -1, 0, -1, 0, 0)
    R55 = Matrix3(-1, 0, 0, 0, -1, 0, 0, 0, -1)
    pose1 = Pose(R=R1, t=Vector3(1, 4, -4))
    pose2 = Pose(R=R2, t=Vector3(2, 6, -12))
    pose3 = Pose(R=R3, t=Vector3(-1, -6, 4))
    pose4 = Pose(R=R4, t=Vector3(-2, -4, 12))
    pose5 = Pose(R=R5, t=Vector3(1, 1, 1))
    pose_gt1 = Pose(R=R11, t=Vector3(-1, -4, 4))
    pose_gt2 = Pose(R=R22, t=Vector3(-2, -6, 12))
    pose_gt3 = Pose(R=R33, t=(1, 6, -4))
    pose_gt4 = Pose(R=R44, t=Vector3(2, 4, -12))
    pose_gt5 = Pose(R=R55, t=Vector3(-1, -1, -1))
    pr_poses = [pose1, pose2, pose3, pose4, pose5]
    gt_poses = [pose_gt1, pose_gt2, pose_gt3, pose_gt4, pose_gt5]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_pose(dir_path, pr_poses, gt_poses)


if __name__ == "__main__":
    main()
