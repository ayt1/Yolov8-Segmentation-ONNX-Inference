import time
import onnxruntime
import numpy as np
import cv2
from skimage.transform import resize as skresize


class YoloV8Seg(object):
    def __init__(self, model_path, conf=0.4, nms_thresh=0.7):
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model_path, providers=self.providers)
        self.conf_thresh = conf
        self.nms_thresh = nms_thresh
        self.data_type = np.half if self.session.get_inputs()[0].type == 'tensor(float16)' else np.float32

    def _letterbox(self, im, new_shape=(640, 640), scaleup=True):
        color = (114, 114, 114)
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im

    def _preprocess_image(self, image):
        img = self._letterbox(image)  # Pad and resize
        img = img.transpose((2, 0, 1))[::-1]  # Convert HWC to CHW, BGR to RGB
        img_in = np.ascontiguousarray(img).astype(self.data_type) / 255
        img_in = img_in[None] if len(img_in) == 3 else img_in

        return img_in

    def _xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def _non_max_suppression(self, prediction, multi_label=False, max_det=300, nc=80):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = np.amax(prediction[:, 4:mi], axis=1) > self.conf_thresh

        # Settings
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000
        time_limit = 10  # seconds to quit after
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        prediction = prediction.transpose((0, 2, 1))
        prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])
        t = time.time()
        output = [np.zeros((0, 6 + nm), dtype=np.float32)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            # If none remain process next image
            if not x.shape[0]:
                continue
            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x[:, :4], x[:, 4:(4 + nc)], x[:, 4 + nc:]

            if multi_label:
                i, j = (x[:, 5:] > self.conf_thresh).nonzero(as_tuple=False).T
                x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype("float32")), 1)
            else:  # best class only
                conf = cls.max(axis=1, keepdims=True)
                j = np.argmax(cls, axis=1).reshape(conf.shape)
                x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(-1) > self.conf_thresh]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.nms_thresh)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]

            output[xi] = x[i]

            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output

    def xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

        Returns:
            y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
        y = np.empty_like(x)  # faster than clone/copy
        dw = x[..., 2] / 2  # half-width
        dh = x[..., 3] / 2  # half-height
        y[..., 0] = x[..., 0] - dw  # top left x
        y[..., 1] = x[..., 1] - dh  # top left y
        y[..., 2] = x[..., 0] + dw  # bottom right x
        y[..., 3] = x[..., 1] + dh  # bottom right y

        return y

    def _scale_mask_coords(self, img1_shape, coords, img0_shape, ratio_pad=None, padding=True):
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        scaled_coords = []
        for coor in coords:
            if padding:
                coor[..., 0] -= pad[0]  # x padding
                coor[..., 1] -= pad[1]  # y padding
            coor[..., 0] /= gain
            coor[..., 1] /= gain

            coor[..., 0] = coor[..., 0].clip(0, img0_shape[1])  # x
            coor[..., 1] = coor[..., 1].clip(0, img0_shape[0])  # y
            scaled_coords.append(coor)

        return scaled_coords

    def _scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2

        return coords

    def _post_processing(self, ori_img, prep_img, output):
        pred = self._non_max_suppression(output[0])[0]
        proto = output[1][0]

        if not len(pred):  # save empty boxes
            masks = None
        else:
            masks = self.process_mask(proto, pred[:, 6:], pred[:, :4], prep_img.shape[2:], upsample=True)  # HWC
            pred[:, :4] = self._scale_coords(prep_img.shape[2:], pred[:, :4], ori_img.shape)
            masks = [self._scale_mask_coords(prep_img.shape[2:], x, ori_img.shape) for x in self.masks2segments(masks)]

        return {"boxes": pred[:, :6], "masks": masks}

    def crop_mask(self, masks, boxes):
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
        c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, shape, upsample=True):
        c, mh, mw = protos.shape  # CHW
        ih, iw = shape
        masks = (1 / (1 + np.exp(-1 * (masks_in @ protos.reshape(c, -1))))).reshape(-1, mh, mw)  # CHW

        downsampled_bboxes = np.copy(bboxes)
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = self.crop_mask(masks, downsampled_bboxes)  # CHW
        if upsample:
            masks = skresize(masks, (masks.shape[0], ih, iw), order=1)
        return np.greater(masks, 0.5)

    def masks2segments(self, masks, strategy='largest'):
        segments = []
        for x in masks.astype('uint8'):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if c:
                segments.append([cont.reshape(-1, 2).astype('float32') for cont in c])
            else:
                segments.append([np.zeros((0, 2)).astype('float32')])

        return segments

    def draw_masks(self, orig_img, seg_result, class_names):
        class_ids = list(seg_result['boxes'][:, -1].astype('int'))
        unique_class_ids = list(set(class_ids))
        class_colors = {class_id: tuple([int(x) for x in np.random.randint(0, 256, 3)]) for class_id in
                        unique_class_ids}
        canvas = np.zeros(orig_img.shape, dtype=np.uint8)
        height, width = orig_img.shape[:2]
        text_size = min([height, width]) * 0.0006
        text_thickness = int(min([height, width]) * 0.001)
        for i, (masks, class_id) in enumerate(zip(seg_result['masks'], class_ids)):
            mask_color = class_colors[class_id]
            for mask in masks:
                cv2.fillPoly(canvas, [mask.astype('int32')], mask_color)
            obj_box = seg_result['boxes'][i][:4].astype('int')
            label = class_names[class_id]
            (tw, th), _ = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size,
                                          thickness=text_thickness)

            cv2.rectangle(canvas, (obj_box[0], obj_box[1]), (obj_box[2], obj_box[3]), mask_color, 1)
            cv2.rectangle(canvas, (obj_box[0], obj_box[1]), (obj_box[0] + tw, obj_box[1] - th), mask_color, -1)
            cv2.putText(canvas, class_names[class_id], (obj_box[0], obj_box[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        text_size, (255, 255, 255), text_thickness, cv2.LINE_AA)
        image_with_mask = cv2.addWeighted(orig_img, 0.8, canvas, 0.6, 0)

        return image_with_mask

    def draw_boxes(self, img, boxes, color):
        height, width = img.shape[:2]
        size = min([height, width]) * 0.0006
        text_thickness = int(min([height, width]) * 0.001)

    def process(self, image):
        # Preprocess
        prep_image = self._preprocess_image(image)
        # Run inference
        input_name = self.session.get_inputs()[0].name
        prediction = self.session.run(None, {input_name: prep_image})
        # Post process
        result = self._post_processing(image, prep_image, prediction)

        return result
