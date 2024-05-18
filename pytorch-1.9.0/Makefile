# This makefile does nothing but delegating the actual building to cmake.

all:
	@mkdir -p build && cd build && cmake .. $(shell python ./scripts/get_python_cmake_flags.py) && $(MAKE)

local:
	@./scripts/build_local.sh

android:
	@./scripts/build_android.sh

ios:
	@./scripts/build_ios.sh

clean: # This will remove ALL build folders.
	@rm -r build*/
	@$(RM) -r $(SHELLCHECK_GHA_GENERATED_FOLDER)

linecount:
	@cloc --read-lang-def=caffe.cloc caffe2 || \
		echo "Cloc is not available on the machine. You can install cloc with " && \
		echo "    sudo apt-get install cloc"

SHELLCHECK_GHA_GENERATED_FOLDER=.shellcheck_generated_gha
shellcheck-gha:
	@$(RM) -r $(SHELLCHECK_GHA_GENERATED_FOLDER)
	tools/extract_scripts.py --out=$(SHELLCHECK_GHA_GENERATED_FOLDER)
	tools/run_shellcheck.sh $(SHELLCHECK_GHA_GENERATED_FOLDER)

generate-gha-workflows:
	./.github/scripts/generate_linux_ci_workflows.py
	$(MAKE) shellcheck-gha

setup_lint:
	python tools/actions_local_runner.py --file .github/workflows/lint.yml \
	 	--job 'flake8-py3' --step 'Install dependencies' --no-quiet
	python tools/actions_local_runner.py --file .github/workflows/lint.yml \
	 	--job 'cmakelint' --step 'Install dependencies' --no-quiet
	python tools/actions_local_runner.py --file .github/workflows/lint.yml \
	 	--job 'mypy' --step 'Install dependencies' --no-quiet

	@if [ "$$(uname)" = "Darwin" ]; then \
		if [ -z "$$(which brew)" ]; then \
			echo "'brew' is required to install ShellCheck, get it here: https://brew.sh "; \
			exit 1; \
		fi; \
		brew install shellcheck; \
	else \
		python tools/actions_local_runner.py --file .github/workflows/lint.yml \
		--job 'quick-checks' --step 'Install ShellCheck' --no-quiet; \
	fi
	pip install jinja2

quick_checks:
	@python tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'quick-checks' \
		--step 'Extract scripts from GitHub Actions workflows'

# TODO: This is broken when 'git config submodule.recurse' is 'true' since the
# lints will descend into third_party submodules
	@python tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'quick-checks' \
		--step 'Ensure no trailing spaces' \
		--step 'Ensure no tabs' \
		--step 'Ensure no non-breaking spaces' \
		--step 'Ensure canonical include' \
		--step 'Ensure no versionless Python shebangs' \
		--step 'Ensure no unqualified noqa' \
		--step 'Ensure no unqualified type ignore' \
		--step 'Ensure no direct cub include' \
		--step 'Run ShellCheck' \
		--step 'Ensure correct trailing newlines'

flake8:
	@python tools/actions_local_runner.py \
		--file-filter '.py' \
		$(CHANGED_ONLY) \
		--job 'flake8-py3'

mypy:
	@python tools/actions_local_runner.py \
		--file-filter '.py' \
		$(CHANGED_ONLY) \
		--job 'mypy'

cmakelint:
	@python tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'cmakelint' \
		--step 'Run cmakelint'

clang_tidy:
	echo "clang-tidy local lint is not yet implemented"
	exit 1

toc:
	@python tools/actions_local_runner.py \
		--file .github/workflows/lint.yml \
		--job 'toc' \
		--step "Regenerate ToCs and check that they didn't change"

lint: flake8 mypy quick_checks cmakelint generate-gha-workflows

quicklint: CHANGED_ONLY=--changed-only
quicklint: mypy flake8 mypy quick_checks cmakelint generate-gha-workflows
