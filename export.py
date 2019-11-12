import time

import torch

from models.deeplab import DeepLab

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model'].module
    print(type(model))

    filename = 'deep_mobile_matting.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))

    print('loading {}...'.format(filename))
    start = time.time()
    model = DeepLab(backbone='mobilenet', output_stride=16, num_classes=1)
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))

    # scripted_model_file = 'deep_mobile_matting_scripted.pt'
    # torch.jit.save(torch.jit.script(model), scripted_model_file)
