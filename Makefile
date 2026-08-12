# ============================================================================
# 本ソフトウェアは実験・研究・個人利用のみを目的とし、無保証で提供される。
# 使用は必ず自身が管理・所有するネットワークに限ること。
# 本ソフトウェアの使用により生じたいかなる損害についても、作者は一切の責任を負わない。
# ============================================================================
#
# 使い方:
#   make            … ローカル GPU のアーキを自動検出してビルド (-arch=native)
#   make ARCH=sm_86 … アーキを固定したい場合
#   make clean      … 生成物を削除
#   make rebuild    … クリーンしてから再ビルド
#
#   Linux で gcc のバージョンが CUDA に未対応と言われたら:
#   make CCBIN=g++-13   （nvcc に -ccbin g++-13 を渡す）

NVCC  := nvcc
# nvcc 11.5+ はローカル GPU の compute capability を自動検出
ARCH  ?= native
CHAP_SRC := chap_md5_bulldozer.cu
SIP_SRC  := sip_digest_bulldozer.cu
CCBIN ?=

UNAME_S := $(shell uname -s 2>/dev/null)
ifeq ($(OS),Windows_NT)
    WINDOWS := 1
endif
ifneq (,$(filter MINGW% MSYS% CYGWIN% Windows%,$(UNAME_S)))
    WINDOWS := 1
endif

ifeq ($(WINDOWS),1)
    CHAP_TARGET := ./build/chap_md5_bulldozer.exe
    SIP_TARGET  := ./build/sip_digest_bulldozer.exe
    HOSTFLAGS   := -Xcompiler=-utf-8,-WX
    WIN_TMP := $(shell cygpath -m /tmp 2>/dev/null || echo C:/Windows/Temp)
    TMPENV  := TMP="$(WIN_TMP)" TEMP="$(WIN_TMP)"
else
    CHAP_TARGET := build/chap_md5_bulldozer
    SIP_TARGET  := build/sip_digest_bulldozer
    HOSTFLAGS   := -Xcompiler=-Wall,-Wextra,-Wno-unknown-pragmas
    TMPENV      :=
endif

NVCCFLAGS := -O3 -arch=$(ARCH) $(HOSTFLAGS) $(if $(CCBIN),-ccbin $(CCBIN),)

.PHONY: all chap sip run clean rebuild

all: $(CHAP_TARGET) $(SIP_TARGET)
chap: $(CHAP_TARGET)
sip: $(SIP_TARGET)

$(CHAP_TARGET): $(CHAP_SRC) | build
	$(TMPENV) $(NVCC) $(NVCCFLAGS) $< -o $@

$(SIP_TARGET): $(SIP_SRC) | build
	$(TMPENV) $(NVCC) $(NVCCFLAGS) $< -o $@

build:
	mkdir -p ./build

clean:
	rm -rf ./build/

rebuild: clean all
