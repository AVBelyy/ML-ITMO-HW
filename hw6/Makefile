all: svd_train baseline/svd_train_baseline

clean:
	rm -f svd_train
	rm -f baseline/svd_train_baseline

svd_train: svd_train.cpp
	g++ -O2 svd_train.cpp -osvd_train -Wno-unused-result

baseline/svd_train_baseline: svd_train.cpp
	g++ -O2 svd_train.cpp -DBASELINE_MODEL -obaseline/svd_train_baseline -Wno-unused-result
