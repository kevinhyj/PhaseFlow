#!/bin/bash
# PhaseFlow Kill Training Script
# 用法: bash scripts/kill_training.sh [option]
# 选项:
#   -f, --force: 强制杀死进程 (SIGKILL)
#   -g, --graceful: 优雅停止 (SIGTERM, 默认)
#   -a, --all: 杀死所有训练进程
#   -p PID, --pid PID: 杀死指定PID的进程

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认使用优雅停止
SIGNAL="TERM"
FORCE=false
KILL_ALL=false
SPECIFIC_PID=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            SIGNAL="KILL"
            FORCE=true
            shift
            ;;
        -g|--graceful)
            SIGNAL="TERM"
            shift
            ;;
        -a|--all)
            KILL_ALL=true
            shift
            ;;
        -p|--pid)
            SPECIFIC_PID="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: bash scripts/kill_training.sh [选项]"
            echo ""
            echo "选项:"
            echo "  -f, --force       强制杀死进程 (SIGKILL)"
            echo "  -g, --graceful    优雅停止 (SIGTERM, 默认)"
            echo "  -a, --all         杀死所有训练进程"
            echo "  -p, --pid PID     杀死指定PID的进程"
            echo "  -h, --help        显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  bash scripts/kill_training.sh              # 优雅停止第一个找到的训练进程"
            echo "  bash scripts/kill_training.sh -f           # 强制杀死第一个找到的训练进程"
            echo "  bash scripts/kill_training.sh -a           # 优雅停止所有训练进程"
            echo "  bash scripts/kill_training.sh -f -a        # 强制杀死所有训练进程"
            echo "  bash scripts/kill_training.sh -p 12345     # 杀死PID为12345的进程"
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "PhaseFlow Kill Training Process"
echo "========================================"

# 如果指定了PID
if [[ -n "$SPECIFIC_PID" ]]; then
    if ps -p "$SPECIFIC_PID" > /dev/null 2>&1; then
        PROCESS_INFO=$(ps -p "$SPECIFIC_PID" -o cmd=)
        echo -e "${YELLOW}找到进程 PID=$SPECIFIC_PID:${NC}"
        echo "$PROCESS_INFO"
        echo ""

        if $FORCE; then
            echo -e "${RED}强制杀死进程 (SIGKILL)...${NC}"
        else
            echo -e "${GREEN}优雅停止进程 (SIGTERM)...${NC}"
        fi

        kill -$SIGNAL "$SPECIFIC_PID"

        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✓ 成功发送信号到进程 $SPECIFIC_PID${NC}"

            # 等待进程结束
            echo "等待进程退出..."
            for i in {1..10}; do
                if ! ps -p "$SPECIFIC_PID" > /dev/null 2>&1; then
                    echo -e "${GREEN}✓ 进程已退出${NC}"
                    exit 0
                fi
                sleep 1
            done

            if ps -p "$SPECIFIC_PID" > /dev/null 2>&1; then
                echo -e "${YELLOW}⚠ 进程仍在运行,可能需要使用 -f 强制杀死${NC}"
                exit 1
            fi
        else
            echo -e "${RED}✗ 无法杀死进程 $SPECIFIC_PID${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ 进程 PID=$SPECIFIC_PID 不存在${NC}"
        exit 1
    fi
fi

# 查找训练进程
echo "搜索训练进程..."
echo ""

# 搜索包含 train.py 或 train.sh 的进程
TRAIN_PIDS=$(pgrep -f "train\.py|train\.sh" 2>/dev/null)

if [[ -z "$TRAIN_PIDS" ]]; then
    echo -e "${YELLOW}⚠ 未找到训练进程${NC}"
    echo ""
    echo "提示: 可以使用以下命令手动查找进程:"
    echo "  ps aux | grep train"
    echo "  ps aux | grep python"
    exit 0
fi

# 显示找到的进程
echo -e "${GREEN}找到以下训练进程:${NC}"
echo ""
printf "%-8s %-8s %-10s %s\n" "PID" "USER" "CPU%" "COMMAND"
echo "----------------------------------------"

for pid in $TRAIN_PIDS; do
    ps -p $pid -o pid=,user=,pcpu=,cmd= | head -1
done

echo ""
echo "========================================"

# 询问确认
if $KILL_ALL; then
    COUNT=$(echo "$TRAIN_PIDS" | wc -w)
    if $FORCE; then
        echo -e "${RED}即将强制杀死 $COUNT 个训练进程 (SIGKILL)${NC}"
    else
        echo -e "${YELLOW}即将优雅停止 $COUNT 个训练进程 (SIGTERM)${NC}"
    fi
else
    FIRST_PID=$(echo "$TRAIN_PIDS" | head -1)
    if $FORCE; then
        echo -e "${RED}即将强制杀死第一个训练进程 PID=$FIRST_PID (SIGKILL)${NC}"
    else
        echo -e "${YELLOW}即将优雅停止第一个训练进程 PID=$FIRST_PID (SIGTERM)${NC}"
    fi
    TRAIN_PIDS=$FIRST_PID
fi

read -p "确认执行? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}已取消${NC}"
    exit 0
fi

echo ""
echo "========================================"

# 杀死进程
SUCCESS_COUNT=0
FAIL_COUNT=0

for pid in $TRAIN_PIDS; do
    if ps -p $pid > /dev/null 2>&1; then
        kill -$SIGNAL $pid 2>/dev/null
        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✓ 成功发送信号到进程 $pid${NC}"
            ((SUCCESS_COUNT++))
        else
            echo -e "${RED}✗ 无法杀死进程 $pid${NC}"
            ((FAIL_COUNT++))
        fi
    else
        echo -e "${YELLOW}⚠ 进程 $pid 已经不存在${NC}"
    fi
done

echo ""
echo "========================================"

# 等待进程结束
if [[ $SUCCESS_COUNT -gt 0 ]]; then
    echo "等待进程退出..."
    sleep 2

    STILL_RUNNING=0
    for pid in $TRAIN_PIDS; do
        if ps -p $pid > /dev/null 2>&1; then
            ((STILL_RUNNING++))
        fi
    done

    if [[ $STILL_RUNNING -eq 0 ]]; then
        echo -e "${GREEN}✓ 所有进程已成功退出${NC}"
    else
        echo -e "${YELLOW}⚠ 仍有 $STILL_RUNNING 个进程在运行${NC}"
        if ! $FORCE; then
            echo -e "${YELLOW}提示: 可以使用 -f 选项强制杀死进程${NC}"
        fi
    fi
fi

echo ""
echo "总结: 成功=$SUCCESS_COUNT, 失败=$FAIL_COUNT"
echo "========================================"

exit 0
