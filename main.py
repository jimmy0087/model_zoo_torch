from modelzoo.train import TRAIN_TEST,parse_opts


opt = parse_opts()
myTest= TRAIN_TEST(opt)
traData,valData = myTest.datasets()
myModel = myTest.model(model_name = 'se_resnet18')
myTest.train(traData,valData,myModel)


