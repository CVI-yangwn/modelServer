#! /bin/bash

LOGPATH=${ROOTPATH}/logs/server_monitor.log

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
cd ${PROJECT_DIR}


    echo -e "${YELLOW}正在检测可用的 Conda 环境...${NC}"
    mapfile -t conda_envs < <(conda env list | grep -v '^#' | awk '{print $1}' | sed '/^$/d')

    if [ ${#conda_envs[@]} -eq 0 ]; then
        echo -e "${RED}错误: 未找到任何 Conda 环境。请先创建 Conda 环境。${NC}"
        exit 1
    fi

    echo -e "${BLUE}发现以下 Conda 环境:${NC}"
    for i in "${!conda_envs[@]}"; do
        printf "  %2d) %s\n" "$((i+1))" "${conda_envs[$i]}"
    done

    local choice
    while true; do
        read -p "请选择要使用的环境序号 [1-${#conda_envs[@]}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#conda_envs[@]}" ]; then
            conda_env_name=${conda_envs[$((choice-1))]}
            echo -e "已选择环境: ${GREEN}${conda_env_name}${NC}"
            break
        else
            echo -e "${RED}输入无效，请输入 1 到 ${#conda_envs[@]} 之间的数字。${NC}"
        fi
    done

source /data/yangwennuo/miniconda3/bin/activate vllm
ulimit -c unlimited
pkill -f trans_server.py
python -u ./trans_server.py > $LOGPATH 2>&1 &

exit 0
