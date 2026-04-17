import random
import numpy as np
import cv2


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """Normalize image by subtracting mean and dividing by std."""
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)
    
    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img


def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)
    
    return pre_img, post_img, label


def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)
    
    return pre_img, post_img, label


def random_scale(pre_img, post_img, label=None, scales=[0.5, 2.0]):
    # scale = random.choice(scales)
    scale = np.random.uniform(scales[0], scales[1])
    if scales != 1:
        sh = int(pre_img.shape[0] * scale)
        # sw = int(pre_img.shape[1] * scale)
        sw = sh
        pre_img = cv2.resize(pre_img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        post_img = cv2.resize(post_img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if label is not None:
        label = cv2.resize(label, (sw, sh), interpolation=cv2.INTER_NEAREST)
    
    return pre_img, post_img, label


def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1
    
    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()
    
    return pre_img, post_img, label


def random_crop(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape
    
    H = max(crop_size, h)
    W = max(crop_size, w)
    
    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)
    
    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index
    
    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]
    
    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]
    
    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))
    
    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label
    
    def get_random_cropbox(cat_max_ratio=0.75):
        
        for i in range(10):
            
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size
            
            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break
        
        return H_start, H_end, W_start, W_end,
    
    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
    
    return pre_img, post_img, label


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.longlong)
        self._epsilon = 1e-7
    
    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc
    
    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + self._epsilon)
        mAcc = np.nanmean(Acc)
        return mAcc, Acc
    
    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Pre = self.confusion_matrix[1, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1] + self._epsilon)
        return Pre
    
    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1] + self._epsilon)
        return Rec
    
    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = 2 * Rec * Pre / (Rec + Pre)
        return F1
    
    def calculate_per_class_metrics(self):
        # Adjustments to exclude class 0 in calculations
        TPs = np.diag(self.confusion_matrix)[1:]  # Start from index 1 to exclude class 0
        FNs = np.sum(self.confusion_matrix, axis=1)[1:] - TPs
        FPs = np.sum(self.confusion_matrix, axis=0)[1:] - TPs
        return TPs, FNs, FPs
    
    def Damage_F1_score(self):
        TPs, FNs, FPs = self.calculate_per_class_metrics()
        precisions = TPs / (TPs + FPs + 1e-7)
        recalls = TPs / (TPs + FNs + 1e-7)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        return f1_scores
    
    def Mean_Intersection_over_Union(self):
        MIoU = np.nanmean(self.Intersection_over_Union())
        return MIoU
    
    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix) + 1e-7)
        return IoU
    
    def Kappa_coefficient(self):
        # Number of observations (total number of classifications)
        # num_total = np.array(0, dtype=np.long)
        # row_sums = np.array([0, 0], dtype=np.long)
        # col_sums = np.array([0, 0], dtype=np.long)
        # total += np.sum(self.confusion_matrix)
        # # Observed agreement (i.e., sum of diagonal elements)
        # observed_agreement = np.sum(np.diag(self.confusion_matrix))
        # # Compute expected agreement
        # row_sums += np.sum(self.confusion_matrix, axis=0)
        # col_sums += np.sum(self.confusion_matrix, axis=1)
        # expected_agreement = np.sum((row_sums * col_sums) / total)
        num_total = np.sum(self.confusion_matrix)
        observed_accuracy = np.trace(self.confusion_matrix) / num_total
        expected_accuracy = np.sum(
            np.sum(self.confusion_matrix, axis=0) / num_total * np.sum(self.confusion_matrix, axis=1) / num_total)
        
        # Calculate Cohen's kappa
        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
        return kappa
    
    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
    
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)