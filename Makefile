# ============================================================================
# 本ソフトウェアは実験・研究・個人利用のみを目的とし、無保証で提供される。
# 使用は必ず自身が管理・所有するネットワークに限ること。
# 本ソフトウェアの使用により生じたいかなる損害についても、作者は一切の責任を負わない。
# ============================================================================
#
# 使い方:
#   make            … ローカル GPU のアーキを自動検出してビルド (-arch=native)
#   make ARCH=sm_86 … アーキを固定したい場合
#   make run        … ビルドして実行
#   make clean      … 生成物を削除
#   make rebuild    … クリーンしてから再ビルド
#
#   Linux で gcc のバージョンが CUDA に未対応と言われたら:
#   make CCBIN=g++-13   （nvcc に -ccbin g++-13 を渡す）

NVCC  := nvcc
ARCH  ?= native          # nvcc 11.5+ はローカル GPU の compute capability を自動検出
SRC   := chap_md5_bulldozer.cu
CCBIN ?=

ifeq ($(OS),Windows_NT)
    TARGET    := chap_md5_bulldozer.exe
    HOSTFLAGS := -Xcompiler=-utf-8,-WX
else
    TARGET    := chap_md5_bulldozer
    HOSTFLAGS := -Xcompiler=-Wall,-Wextra,-Wno-unknown-pragmas
endif

NVCCFLAGS := -O3 -arch=$(ARCH) $(HOSTFLAGS) $(if $(CCBIN),-ccbin $(CCBIN),)

.PHONY: all run clean rebuild

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f chap_md5_bulldozer chap_md5_bulldozer.exe \
	      chap_md5_bulldozer.exp chap_md5_bulldozer.lib

rebuild: clean all
