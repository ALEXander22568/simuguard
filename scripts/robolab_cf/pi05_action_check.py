"""Send 3 fixed observations to a fresh Pi0.5 server and save the returned action chunks."""
import sys
import numpy as np
from openpi_client import websocket_client_policy

out, port = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(0)
client = websocket_client_policy.WebsocketClientPolicy("localhost", port)
chunks = []
for i in range(3):
    obs = {
        "observation/exterior_image_1_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/joint_position": rng.uniform(-1, 1, 7),
        "observation/gripper_position": rng.uniform(0, 1, 1),
        "prompt": "put the banana in the bowl",
    }
    chunks.append(np.asarray(client.infer(obs)["actions"]))
np.save(out, np.stack(chunks))
print(out, np.stack(chunks).shape)
