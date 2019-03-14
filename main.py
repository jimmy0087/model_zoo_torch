from modelzoo.train import TRAIN_TEST,parse_opts


opt = parse_opts()
myTest= TRAIN_TEST(opt)
myData = myTest.datasets()
myModel = myTest.model()
myTest.train(myData,myModel)


