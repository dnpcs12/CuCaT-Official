"""Apply the evaluation fixes CuCaT uses to a DanceCamAnimator-Official checkout.

    python -m evaluation.patch_dca --dca-root D:/path/to/DanceCamAnimator-Official

The paper's DCM metrics were computed with these edits to DanceCamAnimator's evaluation code
(MIT License): files are read and written as UTF-8 (needed on Windows), ``np.int`` becomes ``int``
(removed in NumPy 1.24), feature and LCD computations trim each clip to the shortest of its camera,
motion, and bone-mask lengths, metrics are cast to ``float`` before saving, and the VMD reader takes
each camera key's ``Orthographic`` byte from its own record. Edits are matched exactly; files that
already contain them are left unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EDITS: dict[str, list[tuple[str, str]]] = {
    'scripts/evaluate.py': [
        (
            "    for bone_mask_file in bone_mask_list:\n        with open(bone_mask_file, 'r') as bmf:\n            bone_mask_data = json.load(bmf)\n",
            "    for bone_mask_file in bone_mask_list:\n        with open(bone_mask_file, 'r', encoding='utf-8') as bmf:\n            bone_mask_data = json.load(bmf)\n",
        ),
        (
            "        file_base = os.path.basename(camera_c_file)\n        with open(camera_c_file, 'r') as ccf:\n            camera_c_data = json.load(ccf)\n",
            "        file_base = os.path.basename(camera_c_file)\n        with open(camera_c_file, 'r', encoding='utf-8') as ccf:\n            camera_c_data = json.load(ccf)\n",
        ),
        (
            "    for f_i in tqdm(range(f_cnt)):\n        with open(camera_c_list[f_i], 'r') as ccf:\n            camera_c_data = json.load(ccf)\n",
            "    for f_i in tqdm(range(f_cnt)):\n        with open(camera_c_list[f_i], 'r', encoding='utf-8') as ccf:\n            camera_c_data = json.load(ccf)\n",
        ),
        (
            "            camera_c_data = json.load(ccf)\n        with open(motion_list[f_i], 'r') as mf:\n            motion_data = json.load(mf)\n",
            "            camera_c_data = json.load(ccf)\n        with open(motion_list[f_i], 'r', encoding='utf-8') as mf:\n            motion_data = json.load(mf)\n",
        ),
        (
            "            motion_data = json.load(mf)\n        with open(bone_mask_list[f_i], 'r') as bmf:\n            bone_mask_data = json.load(bmf)\n",
            "            motion_data = json.load(mf)\n        with open(bone_mask_list[f_i], 'r', encoding='utf-8') as bmf:\n            bone_mask_data = json.load(bmf)\n",
        ),
        (
            '            bone_mask_data = json.load(bmf)\n        file_base = os.path.basename(camera_c_list[f_i])\n',
            "            bone_mask_data = json.load(bmf)\n\n        shared_len = min(\n            len(camera_c_data['camera_eye']),\n            len(motion_data['Keypoints3D']),\n            len(bone_mask_data['bone_mask'])\n        )\n        if shared_len == 0:\n            continue\n\n        camera_c_data = {\n            'camera_eye': camera_c_data['camera_eye'][:shared_len],\n            'camera_z': camera_c_data['camera_z'][:shared_len],\n            'camera_y': camera_c_data['camera_y'][:shared_len],\n            'camera_x': camera_c_data['camera_x'][:shared_len],\n            'Fov': camera_c_data['Fov'][:shared_len],\n        }\n        motion_data = {\n            'Keypoints3D': motion_data['Keypoints3D'][:shared_len]\n        }\n        bone_mask_data = {\n            'bone_mask': bone_mask_data['bone_mask'][:shared_len]\n        }\n\n        file_base = os.path.basename(camera_c_list[f_i])\n",
        ),
        (
            '\n    return (diff.dot(diff) + np.trace(sigma1)\n            + np.trace(sigma2) - 2 * tr_covmean)\n\n',
            '\n    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean\n    return float(np.real(fid))\n\n',
        ),
        (
            '        # print(bm_list1[i_bm], bm_list0[i_bm])\n        with open(bm_list1[i_bm], \'r\') as f1:\n            bm1_data = np.array(json.load(f1)["bone_mask"])\n',
            '        # print(bm_list1[i_bm], bm_list0[i_bm])\n        with open(bm_list1[i_bm], \'r\', encoding=\'utf-8\') as f1:\n            bm1_data = np.array(json.load(f1)["bone_mask"])\n',
        ),
        (
            '            bm1_data = np.array(json.load(f1)["bone_mask"])\n        with open(bm_list0[i_bm], \'r\') as f0:\n            bm0_data = np.array(json.load(f0)["bone_mask"])\n',
            '            bm1_data = np.array(json.load(f1)["bone_mask"])\n        with open(bm_list0[i_bm], \'r\', encoding=\'utf-8\') as f0:\n            bm0_data = np.array(json.load(f0)["bone_mask"])\n',
        ),
        (
            '            bm0_data = np.array(json.load(f0)["bone_mask"])\n        bm_frames_cnt += len(bm1_data)\n        lcd += np.sum(np.mean((bm1_data-bm0_data[:len(bm1_data)])**2, axis = 1))\n    lcd /= bm_frames_cnt\n',
            '            bm0_data = np.array(json.load(f0)["bone_mask"])\n        shared_len = min(len(bm1_data), len(bm0_data))\n        if shared_len == 0:\n            continue\n        bm_frames_cnt += shared_len\n        lcd += np.sum(np.mean((bm1_data[:shared_len]-bm0_data[:shared_len])**2, axis = 1))\n    lcd /= bm_frames_cnt\n',
        ),
        (
            "    if os.path.exists(res_metrics_path):\n        with open(res_metrics_path, 'r') as rmf:\n            res_metrics = json.load(rmf)\n",
            "    if os.path.exists(res_metrics_path):\n        with open(res_metrics_path, 'r', encoding='utf-8') as rmf:\n            res_metrics = json.load(rmf)\n",
        ),
        (
            "    if os.path.exists(test_metrics_path):\n        with open(test_metrics_path, 'r') as tmf:\n            test_metrics = json.load(tmf)\n",
            "    if os.path.exists(test_metrics_path):\n        with open(test_metrics_path, 'r', encoding='utf-8') as tmf:\n            test_metrics = json.load(tmf)\n",
        ),
        (
            "\n    with open(res_metrics_path, 'w') as rmwf:\n        json.dump({'fid_k_res':fid_k_res, 'div_k_res':div_k_res,'fid_s_res':fid_s_res, 'div_s_res':div_s_res, 'res_dmr': res_dmr,'lcd': lc_diff},rmwf)\n    with open(test_metrics_path, 'w') as tmwf:\n        json.dump({'div_k_test':div_k_test, 'div_s_test':div_s_test},tmwf)\n    \n",
            "\n    with open(res_metrics_path, 'w', encoding='utf-8') as rmwf:\n        json.dump({'fid_k_res':float(fid_k_res), 'div_k_res':float(div_k_res),'fid_s_res':float(fid_s_res), 'div_s_res':float(div_s_res), 'res_dmr': float(res_dmr),'lcd': float(lc_diff)},rmwf)\n    with open(test_metrics_path, 'w', encoding='utf-8') as tmwf:\n        json.dump({'div_k_test':float(div_k_test), 'div_s_test':float(div_s_test)},tmwf)\n    \n",
        ),
    ],
    'scripts/camera_preprocess.py': [
        (
            '        i_path = os.path.join(input_dir, i_json)\n        o_path = os.path.join(output_dir, i_json)\n        with open(i_path, \'r\') as fin:\n            camera_data = json.load(fin)\n        keyframe_number = camera_data["CameraKeyFrameNumber"]\n',
            '        i_path = os.path.join(input_dir, i_json)\n        o_path = os.path.join(output_dir, i_json)\n        with open(i_path, \'r\', encoding=\'utf-8\') as fin:\n            camera_data = json.load(fin)\n        keyframe_number = camera_data["CameraKeyFrameNumber"]\n',
        ),
        (
            '            # f_i += 1\n\n        with open(o_path,\'w\') as fout:\n            json.dump({\n                "FrameTime":max_frame,\n',
            '            # f_i += 1\n\n        with open(o_path,\'w\', encoding=\'utf-8\') as fout:\n            json.dump({\n                "FrameTime":max_frame,\n',
        ),
        (
            '        i_path = os.path.join(input_dir, i_json)\n        o_path = os.path.join(output_dir, i_json)\n        with open(i_path, \'r\') as fin:\n            camera_data = json.load(fin)\n        max_frame = camera_data["FrameTime"]\n',
            '        i_path = os.path.join(input_dir, i_json)\n        o_path = os.path.join(output_dir, i_json)\n        with open(i_path, \'r\', encoding=\'utf-8\') as fin:\n            camera_data = json.load(fin)\n        max_frame = camera_data["FrameTime"]\n',
        ),
        (
            "            camera_fov.append(fov[i])\n        with open(o_path,'w') as fout:\n            json.dump({\n",
            "            camera_fov.append(fov[i])\n        with open(o_path,'w', encoding='utf-8') as fout:\n            json.dump({\n",
        ),
    ],
    'scripts/my_utils/detect_bone_mask.py': [
        (
            "        motion_gt_path = os.path.join(motion_gt_dir,motion_gt_i)\n        with open(camera_c_path, 'r') as cf:\n            camera_data = json.load(cf)\n",
            "        motion_gt_path = os.path.join(motion_gt_dir,motion_gt_i)\n        with open(camera_c_path, 'r', encoding='utf-8') as cf:\n            camera_data = json.load(cf)\n",
        ),
        (
            "            camera_data = json.load(cf)\n        with open(motion_gt_path, 'r') as mf:\n            motion_data = json.load(mf)\n",
            "            camera_data = json.load(cf)\n        with open(motion_gt_path, 'r', encoding='utf-8') as mf:\n            motion_data = json.load(mf)\n",
        ),
        (
            '\n        bone_mask = (cos_y_z >= cos_fov*np.sqrt(np.sum(kps_yz*kps_yz, axis=-1))).astype(np.int) + (cos_x_z >= cos_fov*np.sqrt(np.sum(kps_xz*kps_xz, axis=-1))).astype(np.int)\n        bone_mask = (bone_mask >= 2).astype(np.int32).transpose(1,0)#\n',
            '\n        bone_mask = (cos_y_z >= cos_fov*np.sqrt(np.sum(kps_yz*kps_yz, axis=-1))).astype(int) + (cos_x_z >= cos_fov*np.sqrt(np.sum(kps_xz*kps_xz, axis=-1))).astype(int)\n        bone_mask = (bone_mask >= 2).astype(np.int32).transpose(1,0)#\n',
        ),
        (
            "        bone_mask_path = os.path.join(bone_mask_dir,bone_mask_name)\n        with open(bone_mask_path, 'w') as bf:\n            json.dump({\n",
            "        bone_mask_path = os.path.join(bone_mask_dir,bone_mask_name)\n        with open(bone_mask_path, 'w', encoding='utf-8') as bf:\n            json.dump({\n",
        ),
    ],
    'scripts/my_utils/vmd.py': [
        (
            '                "ViewAngle": struct.unpack("<I", array[current_index+56: current_index+60])[0],\n                "Orthographic": array[60]\n            })\n',
            '                "ViewAngle": struct.unpack("<I", array[current_index+56: current_index+60])[0],\n                "Orthographic": array[current_index+60]\n            })\n',
        ),
        (
            '            camera_data["CameraKeyFrameRecord"][i]["Rotation"]["z"]))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][0].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][1].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][2].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][3].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list = pad_zero_list(array_list,20)\n            array_list.append(struct.pack("<f",camera_data["CameraKeyFrameRecord"][i]["ViewAngle"]))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Orthographic"].to_bytes(1,byteorder=\'little\', signed=False))\n\n        light_keyframe_number = 0\n        array_list.append(light_keyframe_number.to_bytes(4,byteorder=\'little\', signed=False))\n\n        for i in range(light_keyframe_number):\n',
            '            camera_data["CameraKeyFrameRecord"][i]["Rotation"]["z"]))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][0].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][1].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][2].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][3].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list = pad_zero_list(array_list,20)\n            view_angle = int(round(float(camera_data["CameraKeyFrameRecord"][i]["ViewAngle"])))\n            view_angle = max(1, min(180, view_angle))\n            array_list.append(struct.pack("<I", view_angle))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Orthographic"].to_bytes(1,byteorder=\'little\', signed=False))\n\n        light_keyframe_number = 0\n        array_list.append(light_keyframe_number.to_bytes(4,byteorder=\'little\', signed=False))\n\n        for i in range(light_keyframe_number):\n',
        ),
        (
            '            float(camera_data["CameraKeyFrameRecord"][i]["Rotation"]["z"])))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][0].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][1].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][2].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][3].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list = pad_zero_list(array_list,20)\n            array_list.append(struct.pack("<f",camera_data["CameraKeyFrameRecord"][i]["ViewAngle"]))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Orthographic"].to_bytes(1,byteorder=\'little\', signed=False))\n\n        light_keyframe_number = 0\n        array_list.append(light_keyframe_number.to_bytes(4,byteorder=\'little\', signed=False))\n\n        array = b\'\'.join(array_list)\n',
            '            float(camera_data["CameraKeyFrameRecord"][i]["Rotation"]["z"])))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][0].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][1].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][2].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][3].to_bytes(1,byteorder=\'little\', signed=False))\n            array_list = pad_zero_list(array_list,20)\n            view_angle = int(round(float(camera_data["CameraKeyFrameRecord"][i]["ViewAngle"])))\n            view_angle = max(1, min(180, view_angle))\n            array_list.append(struct.pack("<I", view_angle))\n            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Orthographic"].to_bytes(1,byteorder=\'little\', signed=False))\n\n        light_keyframe_number = 0\n        array_list.append(light_keyframe_number.to_bytes(4,byteorder=\'little\', signed=False))\n\n        array = b\'\'.join(array_list)\n',
        ),
    ],
}


def patch_file(path: Path, edits: list[tuple[str, str]]) -> str:
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    had_final_newline = text.endswith("\n")
    if not had_final_newline:
        text += "\n"
    applied = 0
    for old, new in edits:
        if text.count(old) == 1:
            text = text.replace(old, new)
            applied += 1
        elif new in text:
            continue  # already applied
        else:
            raise RuntimeError(f"{path}: cannot find the code to patch; is this an unmodified checkout?\n{old}")
    if applied == 0:
        return "already patched"
    if not had_final_newline:
        text = text[:-1]
    if crlf:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))
    return f"{applied} edit(s) applied"


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply CuCaT's evaluation fixes to a DanceCamAnimator checkout.")
    parser.add_argument("--dca-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.dca_root.resolve()
    for relative, edits in EDITS.items():
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Not a DanceCamAnimator checkout (missing {relative}): {root}")
        print(f"{relative}: {patch_file(path, edits)}")


if __name__ == "__main__":
    main()
