from modelzoo.libs.train import TRAIN_TEST,parse_opts

opt = parse_opts('./modelzoo/config/opts.json')
myTest = TRAIN_TEST(opt)
traData, valData = myTest.datasets('CIFAR10')
myModel = myTest.model(model_name = 'AlexNet')
myTest.train(traData, valData, myModel)


