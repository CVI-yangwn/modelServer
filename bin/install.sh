#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

TARGET_COMMAND="ms"
SHELL_CONFIG_FILE="$HOME.bashrc"

echo "🚀 开始安装 '${TARGET_COMMAND}' 命令..."

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "项目目录: ${YELLOW}${PROJECT_DIR}${NC}"

echo "赋予 '${TARGET_COMMAND}' 执行权限..."
chmod +x "${PROJECT_DIR}/${TARGET_COMMAND}"

PATH_LINE="export PATH=\"${PROJECT_DIR}:\$PATH\""
if grep -qF "$PATH_LINE" "${SHELL_CONFIG_FILE}"; then
    echo -e "路径已存在于 ${SHELL_CONFIG_FILE} 中，无需操作。"
else
    echo "将项目路径添加到 ${SHELL_CONFIG_FILE}..."
    # 在配置文件末尾添加，并附上注释以便将来识别
    echo -e "\n# Added by ms installer to include model server commands" >> "${SHELL_CONFIG_FILE}"
    echo "$PATH_LINE" >> "${SHELL_CONFIG_FILE}"
    echo -e "${GREEN}路径添加成功！${NC}"
fi

echo -e "\n🎉 ${GREEN}安装成功!${NC}"
echo "请执行以下命令使配置立即生效，或重新打开一个新的终端："
echo -e "  ${YELLOW}source ${SHELL_CONFIG_FILE}${NC}"
echo "之后，你就可以在任何地方使用 '${TARGET_COMMAND}' 命令了。"