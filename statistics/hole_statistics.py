import os
import cv2
from utilities import gather_image_from_dir, get_file_name
from evaluation.statistics import Statistics

if __name__ == "__main__":
    label_dir = r'C:\Users\rytis\Desktop\major_review_disertation\dataForTraining_v3_only_epoch\best_weights\output\label/'
    prediction_dir = r'C:\Users\rytis\Desktop\major_review_disertation\dataForTraining_v3_only_epoch\best_weights\output\prediction/'

    label_paths = gather_image_from_dir(label_dir)
    label_paths.sort()
    prediction_paths = gather_image_from_dir(prediction_dir)
    prediction_paths.sort()

    architecture_count = int(float(len(prediction_paths)) / float(len(label_paths)))

    for architecture_index in range(architecture_count):
        print(50 * '*')
        tp_array = []
        fp_array = []
        tn_array = []
        fn_array = []

        recall_array = []
        precision_array = []
        accuracy_array = []
        f1_array = []
        IoU_array = []
        dice_array = []
        mcc_array = []

        first_picture_name = get_file_name(prediction_paths[len(label_paths) * architecture_index])
        print(first_picture_name)
        for i, label_path in enumerate(label_paths):
            prediction_index = i + len(label_paths) * architecture_index
            prediction = cv2.imread(prediction_paths[prediction_index], cv2.IMREAD_GRAYSCALE)
            _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)
            label = cv2.imread(label_paths[i], cv2.IMREAD_GRAYSCALE)
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)

            tp, fp, tn, fn = Statistics.GetParameters(label, prediction)
            recall = Statistics.GetRecall(tp, fn)
            precision = Statistics.GetPrecision(tp, fp)
            accuracy = Statistics.GetAccuracy(tp, fp, tn, fn)
            f1 = Statistics.GetF1Score(recall, precision)
            IoU = Statistics.GetIoU(label, prediction)
            dice = Statistics.GetDiceCoef(label, prediction)
            mcc = Statistics.GetMCC(tp, fp, tn, fn)

            tp_array.append(tp)
            fp_array.append(fp)
            tn_array.append(tn)
            fn_array.append(fn)

            recall_array.append(recall)
            precision_array.append(precision)
            accuracy_array.append(accuracy)
            f1_array.append(f1)
            IoU_array.append(IoU)
            dice_array.append(dice)
            mcc_array.append(mcc)

        sum_recall = float(sum(recall_array))
        sum_precision = float(sum(precision_array))
        sum_accuracy = float(sum(accuracy_array))
        sum_f1 = float(sum(f1_array))
        sum_IoU = float(sum(IoU_array))
        sum_dice = float(sum(dice_array))
        sum_mcc = float(sum(mcc_array))

        pic_count = float(len(label_paths))
        avg_recall = sum_recall / pic_count
        avg_precision = sum_precision / pic_count
        avg_accuracy = sum_accuracy / pic_count
        avg_f1 = sum_f1 / pic_count
        avg_IoU = sum_IoU / pic_count
        avg_dice = sum_dice / pic_count
        avg_mcc = sum_mcc / pic_count

        tp_sum = sum(f1_array)
        fp_sum = sum(fp_array)
        tn_sum = sum(tn_sum)
        fn_sum = sum(fn_sum)

        print(f'acc: {avg_accuracy}')
        print(f'recall: {avg_recall}')
        print(f'precision: {avg_precision}')
        print(f'f1: {avg_f1}')
        print(f'IoU: {avg_IoU}')
        print(f'dice: {avg_dice}')
        print(f'mcc: {mcc}')
        print(f'tp: {tp}')
        print(f'fp: {fp}')
        print(f'tn: {tn}')
        print(f'fn: {fn}')

        # cv2.imshow('label', label)
        # cv2.imshow('prediction', prediction)
        # cv2.waitKey(1)
