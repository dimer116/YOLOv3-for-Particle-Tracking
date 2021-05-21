import os
import SimulateTool
import numpy as np
import sys

# Generates images that are saved in training_data.
# Run with integer parameters for starting index, and number of iterations.

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
args = sys.argv
Starting_index = int(args[1])
iterations = int(args[2])
print("Starting index:", Starting_index, "Iterations:", iterations)

Generate_images = True
Update_csv = False
Only_generate_test_images = False

if Update_csv:
    train_override = open("train.csv", "w")
    train_override.write("img_file,label_file\n")
    train_override.close()
    test_override = open("test.csv", "w")
    test_override.write("img_file,label_file\n")
    test_override.close()

    train_file = open("train.csv", "a")
    test_file = open("test.csv", "a")
    for i in range(Starting_index, Starting_index + iterations):
        if i % 20 == 5:
            test_file.write("img" + str(i) + ".npy,boxes" + str(i) + ".npy\n")
        else:
            train_file.write("img" + str(i) + ".npy,boxes" + str(i) + ".npy\n")
    test_file.close()
    train_file.close()

if Generate_images:
    simulator = SimulateTool.SimulateTool(numImages=0, addStructNoise=False, numParticlesRange=[0, 10], addNoise=True,
                                          addAugment=False,
                                          addIllumination=True, imageSize=[416, 416])
    for i in range(Starting_index, Starting_index + iterations):
        if Only_generate_test_images:
            if i % 20 != 5:
                print(str(i))
                continue
        im = simulator.getNewImage()
        orig_bboxes = simulator.get_all_pos(im)

        np.save('../training_data/images/img' + str(i), im)
        np.save('../training_data/labels/boxes' + str(i), orig_bboxes)
        print('iteration: ' + str(i))
