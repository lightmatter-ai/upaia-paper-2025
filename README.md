## Data and code for <i>Universal Photonic Artificial Intelligence Acceleration</i> paper

### ML experiments
| Model                        | Dataset                                                                 | Checkpoint                                                                 |
|------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Resnet18                     | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)                 | [Google's cifar10 checkpoint](https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth), [Backup](./resnet18/backup_google_cifar10.pth) |
| Resnet18                     | [Imagenette](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz)    | [Fastai](https://github.com/fastai/imagenette)                               |
| Resnet18                     | [Imagewoof](https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz)     | [Fastai](https://github.com/fastai/imagenette)                               |
| 3-layer FFN (19K params)     | [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)          | [link](./mlp/mnist.pth)                       |
| 4-layer FFN (240K params)    | [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) | [link](./mlp/fmnist.pth)                       |
| Atari-DQN                    | [CleanRL](https://github.com/vwxyzjn/cleanrl)                               | [HuggingFace](https://huggingface.co/cleanrl)                          |
| Traffic Light                | [Traffic light](./traffic_light) | [extracted from ApolloAuto](./trafficlight/vertical_weights_pytorch.pt), see [link](https://github.com/ApolloAuto/apollo/tree/master/modules/perception/traffic_light_detection/data)  |
| Bert-tiny / IMDB             | [IMDB](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)          | [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)                  |
| Bert-tiny / Squad            | [Squad](https://rajpurkar.github.io/SQuAD-explorer/)                    | [mrm8488/bert-small-finetuned-squadv2](https://huggingface.co/mrm8488/bert-small-finetuned-squadv2) |
| SegNet / IIIT Pets           | [IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/)                | [link](./iit-pets/best-pet-segnet.pth)                                 |
| Tiny Shakespeare             | [Tiny Shakespeare](https://github.com/karpathy/nanoGPT/tree/master/data)            | [link](https://drive.google.com/file/d/1GKNR34HUJFLWVwKEsenhXEWqjy950Aws/)                                 |

## figure 3 data
[folder link](./figure_data/)

