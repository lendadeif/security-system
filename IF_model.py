import os
for file in os.listdir("captures"):
    os.remove(os.path.join("captures", file))