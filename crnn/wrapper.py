import torch
from torch.autograd import Variable
from PIL import Image
import crnn
import cv2


class CRNNWrapper:
    def __init__(self, model_path, alphabet):
        self.model_path = model_path
        self.alphabet = alphabet
        self.model = crnn.models.CRNN(32, 1, 37, 256)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.converter = crnn.utils.strLabelConverter(alphabet)

        print('loading pretrained model from %s' % model_path)
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image).convert('L')
        transformer = crnn.dataset.resizeNormalize((100, 32))
        image = transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        self.model.eval()
        preds = self.model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        print('%-20s => %-20s' % (raw_pred, sim_pred))
        return sim_pred


if __name__ == '__main__':
    model_path = './data/crnn.pth'
    model_path = '/ssd_scratch/cvit/jerin/acl-workspace/crnn.pth'
    img_path = './data/demo.png'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    wrapper = CRNNWrapper(model_path, alphabet)
    # image = Image.open(img_path).convert('L')
    image = cv2.imread(img_path)
    print(wrapper.predict(image))


