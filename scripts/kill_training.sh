#!/bin/bash
# PhaseFlow Kill Training Script
# 用法: bash scripts/kill_training.sh

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SIGNAL="TERM"
FORCE=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            SIGNAL="KILL"
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "用法: bash scripts/kill_training.sh [-f]"
            echo ""
            echo "选项:"
            echo "  -f, --force    强制杀死 (SIGKILL)"
            echo "  -h, --help     显示帮助"
            echo ""
            echo "交互选择:"
            echo "  1        选择第1个"
            echo "  1,3      选择第1和第3个"
            echo "  1-3      选择第1到3个"
            echo "  a        全选"
            echo "  q        退出"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

echo "========================================"
echo -e "${BOLD}PhaseFlow Kill Training${NC}"
echo "========================================"
echo ""

# 查找 PhaseFlow 相关的 python 训练进程
PIDS=()
CMDS=()
TIMES=()

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    pid=$(echo "$line" | awk '{print $1}')
    time=$(echo "$line" | awk '{print $2}')
    cmd=$(echo "$line" | cut -d' ' -f3-)

    # 跳过 grep 自身和这个脚本
    if echo "$cmd" | grep -q "kill_training.sh"; then
        continue
    fi

    PIDS+=("$pid")
    TIMES+=("$time")
    CMDS+=("$cmd")
done < <(ps -eo pid,lstart,cmd --no-headers | grep -E "PhaseFlow.*train|train.*PhaseFlow" | grep -v grep)

# 如果没找到，尝试更宽松的匹配
if [[ ${#PIDS[@]} -eq 0 ]]; then
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        pid=$(echo "$line" | awk '{print $1}')
        time=$(echo "$line" | awk '{print $2}')
        cmd=$(echo "$line" | cut -d' ' -f3-)

        if echo "$cmd" | grep -q "kill_training.sh"; then
            continue
        fi

        PIDS+=("$pid")
        TIMES+=("$time")
        CMDS+=("$cmd")
    done < <(ps -eo pid,lstart,cmd --no-headers | grep -E "python.*train\.py" | grep -v grep)
fi

if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo -e "${YELLOW}未找到训练进程${NC}"
    echo ""
    echo "提示: ps aux | grep train"
    exit 0
fi

# 显示进程列表
echo -e "${GREEN}找到 ${#PIDS[@]} 个训练进程:${NC}"
echo ""

for i in "${!PIDS[@]}"; do
    idx=$((i+1))
    pid="${PIDS[$i]}"

    # 获取详细信息
    info=$(ps -p "$pid" -o %cpu,%mem,etime --no-headers 2>/dev/null)
    cpu=$(echo "$info" | awk '{print $1}')
    mem=$(echo "$info" | awk '{print $2}')
    etime=$(echo "$info" | awk '{print $3}')

    # 截取命令关键部分
    cmd="${CMDS[$i]}"
    # 提取配置文件或关键参数
    config=$(echo "$cmd" | grep -oE "config[s]?/[^ ]+|--[a-z_]+=[^ ]+" | head -2 | tr '\n' ' ')
    short_cmd=$(echo "$cmd" | sed 's|.*/||' | head -c 60)

    echo -e "${CYAN}[$idx]${NC} PID:${BOLD}$pid${NC}  运行:${etime}  CPU:${cpu}%  MEM:${mem}%"
    echo "    $short_cmd"
    [[ -n "$config" ]] && echo -e "    ${YELLOW}$config${NC}"
    echo ""
done

echo "========================================"

# 交互选择
echo -e "选择要杀死的进程 (${CYAN}1${NC}, ${CYAN}1,3${NC}, ${CYAN}1-3${NC}, ${CYAN}a${NC}=全选, ${CYAN}q${NC}=退出):"
read -p "> " selection

[[ "$selection" == "q" || "$selection" == "quit" ]] && echo "已取消" && exit 0

# 解析选择
SELECTED=()
max=${#PIDS[@]}

if [[ "$selection" == "a" || "$selection" == "all" ]]; then
    SELECTED=("${PIDS[@]}")
else
    IFS=',' read -ra parts <<< "$selection"
    for part in "${parts[@]}"; do
        part=$(echo "$part" | tr -d ' ')
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            for ((j=${BASH_REMATCH[1]}; j<=${BASH_REMATCH[2]}; j++)); do
                [[ $j -ge 1 && $j -le $max ]] && SELECTED+=("${PIDS[$((j-1))]}")
            done
        elif [[ "$part" =~ ^[0-9]+$ ]]; then
            [[ $part -ge 1 && $part -le $max ]] && SELECTED+=("${PIDS[$((part-1))]}")
        fi
    done
fi

if [[ ${#SELECTED[@]} -eq 0 ]]; then
    echo -e "${RED}无效选择${NC}"
    exit 1
fi

# 去重
SELECTED=($(printf '%s\n' "${SELECTED[@]}" | sort -u))

echo ""
echo -e "将杀死 ${#SELECTED[@]} 个进程: ${SELECTED[*]}"
if $FORCE; then
    echo -e "${RED}模式: 强制 (SIGKILL)${NC}"
else
    echo -e "${YELLOW}模式: 优雅 (SIGTERM)${NC}"
fi

read -p "确认? (y/N): " -n 1 -r
echo

[[ ! $REPLY =~ ^[Yy]$ ]] && echo "已取消" && exit 0

# 执行
echo ""
for pid in "${SELECTED[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
        kill -$SIGNAL "$pid" 2>/dev/null
        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✓${NC} 已发送信号到 PID $pid"
        else
            echo -e "${RED}✗${NC} 无法杀死 PID $pid"
        fi
    else
        echo -e "${YELLOW}⚠${NC} PID $pid 已不存在"
    fi
done

# 等待
echo ""
echo -n "等待退出"
for i in {1..5}; do
    sleep 1
    echo -n "."
    all_dead=true
    for pid in "${SELECTED[@]}"; do
        ps -p "$pid" > /dev/null 2>&1 && all_dead=false
    done
    $all_dead && break
done
echo ""

# 检查结果
still_running=()
for pid in "${SELECTED[@]}"; do
    ps -p "$pid" > /dev/null 2>&1 && still_running+=("$pid")
done

if [[ ${#still_running[@]} -eq 0 ]]; then
    echo -e "${GREEN}✓ 所有进程已退出${NC}"
else
    echo -e "${YELLOW}⚠ 仍在运行: ${still_running[*]}${NC}"
    $FORCE || echo "提示: 使用 -f 强制杀死"
fi
