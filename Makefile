.PHONY: all proto build test clean install-icd

# Top-level build orchestration
# make all        - build everything
# make proto      - regenerate protobuf code
# make build      - build Go + C components
# make test       - run all tests
# make clean      - clean build artifacts
# make install-icd - install OpenCL ICD registration

GO_BUILD_FLAGS := -ldflags="-s -w"
PROTO_DIR := proto/distri/v1
OUT_DIR := build

all: proto build

# ── Protobuf ────────────────────────────────────────────
proto:
	@echo "Generating protobuf code..."
	buf generate proto/

proto-go:
	protoc --go_out=. --go-grpc_out=. \
		--proto_path=proto \
		proto/distri/v1/control.proto \
		proto/distri/v1/capability.proto \
		proto/distri/v1/compute.proto

# ── Go components ───────────────────────────────────────
build-go: proto-go
	go build $(GO_BUILD_FLAGS) -o $(OUT_DIR)/distribox.exe ./cmd/distribox
	go build $(GO_BUILD_FLAGS) -o $(OUT_DIR)/distribox-cli ./cmd/cli

# ── C/C++ components ────────────────────────────────────
build-engine:
	@mkdir -p $(OUT_DIR)
	cmake -B $(OUT_DIR)/engine -S engine -DCMAKE_BUILD_TYPE=Release
	cmake --build $(OUT_DIR)/engine --config Release

build-icd:
	@mkdir -p $(OUT_DIR)
	cmake -B $(OUT_DIR)/icd -S icd -DCMAKE_BUILD_TYPE=Release
	cmake --build $(OUT_DIR)/icd --config Release

build-c: build-engine build-icd

# ── Combined build ──────────────────────────────────────
build: build-go build-c

# ── Platform-specific ───────────────────────────────────
linux-amd64:
	GOOS=linux GOARCH=amd64 $(MAKE) build-go
	cmake -B $(OUT_DIR)/engine-linux -S engine \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-linux-amd64.cmake
	cmake --build $(OUT_DIR)/engine-linux

windows-x64:
	GOOS=windows GOARCH=amd64 $(MAKE) build-go
	cmake -B $(OUT_DIR)/engine-win -S engine \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-windows-x64.cmake
	cmake --build $(OUT_DIR)/engine-win

android-arm64:
	@if [ -f ./scripts/build_android.sh ]; then \
		./scripts/build_android.sh; \
	else \
		powershell -ExecutionPolicy Bypass -File scripts/build_android.ps1; \
	fi

# ── Distribution packaging ──────────────────────────────
dist-windows: build
	@echo "Packaging Windows distribution..."
	@mkdir -p $(OUT_DIR)/dist-windows
	cp $(OUT_DIR)/distribox.exe $(OUT_DIR)/dist-windows/
	cp $(OUT_DIR)/distribox-cli.exe $(OUT_DIR)/dist-windows/ 2>/dev/null || true
	cp $(OUT_DIR)/icd/distribox_icd.dll $(OUT_DIR)/dist-windows/ 2>/dev/null || true
	cp $(OUT_DIR)/engine/distribox_engine.dll $(OUT_DIR)/dist-windows/ 2>/dev/null || true
	cp $(OUT_DIR)/cuda/nvcuda.dll $(OUT_DIR)/dist-windows/ 2>/dev/null || true
	cp $(OUT_DIR)/vk/distribox_vk_layer.dll $(OUT_DIR)/dist-windows/ 2>/dev/null || true
	cp $(OUT_DIR)/vk/distribox_vk_layer.json $(OUT_DIR)/dist-windows/ 2>/dev/null || true
	cp $(OUT_DIR)/hook/distribox_hook.dll $(OUT_DIR)/dist-windows/ 2>/dev/null || true
	cp $(OUT_DIR)/gl_proxy/distri_opengl32.dll $(OUT_DIR)/dist-windows/ 2>/dev/null || true
	@echo "Distribution: $(OUT_DIR)/dist-windows/"

dist-android:
	@echo "Building Android APK distribution..."
	@if [ -f ./scripts/build_android.sh ]; then \
		./scripts/build_android.sh; \
	else \
		powershell -ExecutionPolicy Bypass -File scripts/build_android.ps1; \
	fi

dist: dist-windows
	@echo "All distributions built!"

# ── Test ────────────────────────────────────────────────
test: test-go test-c

test-go:
	go test -v -race ./pkg/... ./cmd/... ./vgpu/...

test-c:
	cd $(OUT_DIR)/engine && ctest --output-on-failure
	cd $(OUT_DIR)/icd && ctest --output-on-failure

# ── ICD registration ────────────────────────────────────
install-icd-linux:
	@echo "Installing ICD for Linux..."
	@mkdir -p /etc/OpenCL/vendors
	@echo "$(PWD)/$(OUT_DIR)/icd/libdistribox_icd.so" > /etc/OpenCL/vendors/distribox.icd
	@echo "ICD installed. Run 'make uninstall-icd-linux' to remove."

install-icd-win:
	@echo "Installing ICD for Windows (requires admin)..."
	powershell -ExecutionPolicy Bypass -File scripts/install_icd.ps1

# ── Clean ───────────────────────────────────────────────
clean:
	rm -rf $(OUT_DIR)
	go clean -cache

# ── Help ────────────────────────────────────────────────
help:
	@echo "DistriBox Build System"
	@echo "  make all              Build everything"
	@echo "  make proto            Generate protobuf"
	@echo "  make build            Build Go + C"
	@echo "  make build-go         Build Go components only"
	@echo "  make build-c          Build C components only"
	@echo "  make test             Run all tests"
	@echo "  make install-icd-linux  Register ICD on Linux"
	@echo "  make android-arm64    Cross-compile for Android"
