ifndef LINT_LANG
	LINT_LANG="all"
endif

ifndef NPROC
	NPROC=1
endif
lint:
	python3 dmlc-core/scripts/lint.py treelite $(LINT_LANG) include src python \
	--exclude_path python/treelite/gallery/sklearn --pylint-rc $(PWD)/python/.pylintrc

doxygen:
	cd docs; doxygen

cpp-coverage:
	rm -rf build; mkdir build; cd build; cmake .. -DTEST_COVERAGE=ON && make -j$(NPROC)

all:
	rm -rf build; mkdir build; cd build; cmake .. && make -j$(NPROC)
