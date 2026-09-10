"""命令行人脸注册（服务器本机有摄像头时用）。
浏览器页面上的「注册」按钮更方便（浏览器即感知设备），本脚本仅作备用。

用法: python register_face.py --name 存孝
      对着摄像头按空格拍 3 张，写入 face_db.json
"""
import argparse
import base64

import cv2

from pipeline import FacePipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="注册人姓名")
    args = ap.parse_args()

    pipe = FacePipe()
    if not pipe.ok:
        raise SystemExit("FacePipe 未就绪（检查 models/face 下的 YuNet/SFace 模型）")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("打不开本机摄像头，改用浏览器页面上的注册按钮吧")

    print("按 [空格] 拍照，共 3 张；按 [q] 退出")
    taken = 0
    while taken < 3:
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imshow("register", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            ok_flag, jpg = cv2.imencode(".jpg", frame)
            if ok_flag:
                res = pipe.register(args.name, base64.b64encode(jpg.tobytes()).decode(), pose="front")
                if res.get("ok"):
                    taken += 1
                    print(f"已拍 {taken}/3 质量 {res.get('quality')}")
                else:
                    print(f"注册失败：{res.get('error', 'unknown')}")
            else:
                print("编码失败")
    cap.release()
    cv2.destroyAllWindows()
    print(f"完成，{args.name} 已注册 {taken} 张 -> face_db.json")


if __name__ == "__main__":
    main()
