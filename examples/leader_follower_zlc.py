#!/usr/bin/env python3
"""Leader/Follower via ZeroLanCom."""

import time
import pyzlc

from franka_control_client.core.message import FrankaResponseCode
from franka_control_client.core.latest_msg_subscriber import LatestMsgSubscriber


ZLC_IP = "141.3.53.63"
GROUP = "224.0.0.1"
GROUP_PORT = 7720
GROUP_NAME = "RobotControlGroup"

LEADER_NODE = "FrankaLeader"
FOLLOWER_NODE = "FrankaFollower"
# Use a unique command topic to avoid accidental multi-publisher conflicts.
FOLLOWER_CMD_TOPIC = f"{FOLLOWER_NODE}/joint_position_command_leaderfollower"

# IMPORTANT:
# The C++ proxy dispatches subscriber handlers on a ~100ms periodic task (≈10 Hz).
# If we publish much faster (e.g. 100 Hz), the follower can fall behind and keep
# executing an old command backlog. So we publish at ~10 Hz to stay real-time.
DT_S = 0.1  # 10 Hz
ALIGN_CALL_TIMEOUT_S = 60.0

PRIME_TICKS = 3

# --------------------
# Gripper sync
# --------------------
GRIPPER_SPEED = 0.5

LEADER_GRIPPER_STATE_TOPIC = "FrankaLeaderGripper/franka_gripper_state"
FOLLOWER_GRIPPER_CMD_TOPIC = "FrankaFollowerGripper/franka_gripper_command"


def _init_zlc() -> None:
    pyzlc.init(
        "LeaderFollower",
        ZLC_IP,
        group=GROUP,
        group_port=GROUP_PORT,
        group_name=GROUP_NAME,
    )

def _wait_for_service(node_name: str, service_suffix: str, timeout_s: float = 10.0) -> str:
    svc = f"{node_name}/{service_suffix}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = pyzlc.check_node_info(node_name)
        if info:
            for s in info.get("services") or []:
                if isinstance(s, dict) and s.get("name") == svc:
                    return svc
        pyzlc.sleep(0.1)
    raise RuntimeError(f"Service not found (timeout {timeout_s:.1f}s): {svc}")


def _wait_for_first_msg(sub: LatestMsgSubscriber, timeout_s: float = 10.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if sub.last_message is not None:
            return sub.last_message
        pyzlc.sleep(DT_S)
    raise RuntimeError(f"Timeout waiting for topic '{sub.topic_name}'")


def main() -> None:
    _init_zlc()

    # Require both proxies to be online before we proceed.
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if pyzlc.check_node_info(LEADER_NODE) and pyzlc.check_node_info(FOLLOWER_NODE):
            break
        # Use pyzlc.sleep() so the LanCom/heartbeat tasks keep running smoothly.
        pyzlc.sleep(0.1)
    else:
        raise RuntimeError(
            "Leader/Follower not discovered. Start both proxies and ensure they use "
            f"group={GROUP} group_port={GROUP_PORT} group_name={GROUP_NAME}."
        )

    leader_state_sub = LatestMsgSubscriber(f"{LEADER_NODE}/franka_arm_state")
    follower_state_sub = LatestMsgSubscriber(f"{FOLLOWER_NODE}/franka_arm_state")
    _wait_for_first_msg(leader_state_sub)
    _wait_for_first_msg(follower_state_sub)

    gripper_state_sub = None
    gripper_cmd_pub = None

    gripper_state_sub = LatestMsgSubscriber(LEADER_GRIPPER_STATE_TOPIC)
    gripper_cmd_pub = pyzlc.Publisher(FOLLOWER_GRIPPER_CMD_TOPIC)
    _wait_for_first_msg(gripper_state_sub)
    w0 = float(gripper_state_sub.get_latest()["width"])
    for _ in range(PRIME_TICKS):
        gripper_cmd_pub.publish({"width": w0, "speed": GRIPPER_SPEED})
        pyzlc.sleep(DT_S)

    leader_svc = _wait_for_service(LEADER_NODE, "set_franka_arm_control_mode", timeout_s=10.0)
    pyzlc.call(leader_svc, "GravityComp")
    print("[INFO] leader: GravityComp")

    # Use a dedicated publisher so we can bind to a unique command topic.
    cmd_pub = pyzlc.Publisher(FOLLOWER_CMD_TOPIC)
    cmd_pub.publish({"pos": follower_state_sub.get_latest()["q"]})
    pyzlc.sleep(0.1)

    try:
        # --------------------
        # ALIGN
        # --------------------
        # Stop any follower control thread before motion-generator move.
        pyzlc.call(f"{FOLLOWER_NODE}/set_franka_arm_control_mode", "Idle")

        q_align = leader_state_sub.get_latest()["q"]
        header, _ = pyzlc.call(
            f"{FOLLOWER_NODE}/move_franka_arm_to_joint_position",
            q_align,
            ALIGN_CALL_TIMEOUT_S,
        )
        if header != FrankaResponseCode.SUCCESS.value:
            raise RuntimeError(f"align service failed: header={header}")
        print("[INFO] follower aligned to leader snapshot")

        # --------------------
        # FOLLOW
        # --------------------
        # Prime the command topic with a fresh target before switching modes.
        # (Some setups can briefly apply an old desired position right after the switch.)
        for _ in range(PRIME_TICKS):
            cmd_pub.publish({"pos": q_align})
            pyzlc.sleep(DT_S)

        pyzlc.call(f"{FOLLOWER_NODE}/set_franka_arm_control_mode", "HybridJointImpedance")
        print(f"[INFO] follower: HybridJointImpedance (following at {1.0/DT_S:.1f} Hz)")

        # Prime again right after the switch (now the subscriber is active).
        for _ in range(PRIME_TICKS):
            cmd_pub.publish({"pos": q_align})
            pyzlc.sleep(DT_S)

        while True:
            cmd_pub.publish({"pos": leader_state_sub.get_latest()["q"]})

            if gripper_state_sub is not None and gripper_cmd_pub is not None:
                msg = gripper_state_sub.last_message
                if msg["width"] < 0.05:
                    gripper_cmd_pub.publish(
                        {"width": float(0.0175), "speed": GRIPPER_SPEED}
                    )
                else:
                    gripper_cmd_pub.publish(
                        {"width": float(0.05), "speed": GRIPPER_SPEED}
                    )        

            pyzlc.sleep(DT_S)

    except KeyboardInterrupt:
        pass
    finally:
        pyzlc.shutdown()


if __name__ == "__main__":
    main()
