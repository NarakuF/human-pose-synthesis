from skimage import io, transform
import numpy as np

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        img = sample['pose']
        parsing = sample['parsing']
        raw = sample['raw']
        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(img, (new_h, new_w))
        parsing = transform.resize(parsing, (new_h, new_w))
        raw = transform.resize(raw, (new_h, new_w))
        sample = {'raw': raw, 'parsing': parsing, 'pose': img, 'annotate': sample['annotate']}
        return sample
 
class DynamicCrop(object):
    def __init__(self, padding):
        assert(isinstance(padding, int))
        self.padding = padding
        return
    
    def __call__(self, sample):
        img = sample['pose']
        parsing = sample['parsing']
        raw = sample['raw']
        h, w = img.shape[:2]
        up, left = float('inf'), float('inf')
        down, right = 0, 0
        for i in range(3):
            non_zero = np.argwhere(img[:,:,i]>0)
            row = [n[0] for n in non_zero]
            col = [n[1] for n in non_zero]
            up, down = min(min(row),up), max(max(row),down)
            left, right = min(min(col),left), max(max(col),right)        

        new_h, new_w = down-up, right-left
        new_up = max(0, up - self.padding)
        new_down = min(h, new_up + new_h + 2*self.padding)
        new_left = max(0, left - self.padding)
        new_right = min(w, new_left + new_w + 2*self.padding)
        img = img[new_up:new_down, new_left:new_right]
        raw = raw[new_up:new_down, new_left:new_right]
        parsing = parsing[new_up:new_down, new_left:new_right]
        sample = {'raw': raw, 'parsing': parsing, 'pose': img, 'annotate': sample['annotate']}
            
        return sample

class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, sample):
        img = sample['pose']
        parsing = sample['parsing']
        raw = sample['raw']
        
        h, w = img.shape[:2]
        non_zero = np.argwhere(img[:,:,1]>0)
        row = [n[0] for n in non_zero]
        col = [n[1] for n in non_zero]
        up, down = min(row), max(row)
        left, right = min(col), max(col)        
        center_h, center_w = (up+down)//2, (left+right)//2
        new_h, new_w = self.output_size

        if center_h - new_h//2 > 0 and center_h + new_h//2 < h and center_w - new_w//2 > 0 and center_w + new_w//2 < w:
            img = img[center_h - new_h//2: center_h + new_h//2,
                      center_w - new_w//2: center_w + new_w//2]
            raw = raw[center_h - new_h//2: center_h + new_h//2,
                      center_w - new_w//2: center_w + new_w//2]
            parsing = parsing[center_h - new_h//2: center_h + new_h//2,
                              center_w - new_w//2: center_w + new_w//2]
        else:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            while top + new_h > h or left + new_w > w:
                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)
            img = img[top: top + new_h,
                      left: left + new_w]
            raw = raw[top: top + new_h,
                      left: left + new_w]
            parsing = parsing[top: top + new_h,
                              left: left + new_w]
        sample = {'raw': raw, 'parsing': parsing, 'pose': img, 'annotate': sample['annotate']}
        return sample
    
    
class ToTensor(object):
    def __call__(self, sample):
        raw, parsing, pose = sample['raw'], sample['parsing'], sample['pose']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        raw = raw.transpose((2, 0, 1))
        parsing = parsing.transpose((2, 0, 1))
        pose = pose.transpose((2, 0, 1))
        sample = {'raw': torch.from_numpy(raw), 'parsing': torch.from_numpy(parsing), 'pose': torch.from_numpy(pose), 'annotate': sample['annotate']}
        return sample