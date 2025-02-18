# Future Pair Trading

本项目实现了期货配对交易的策略分析和数据可视化，旨在帮助用户使用 Python 对多个期货合约的数据进行预处理、分析，并通过图表进行结果展示。

## 功能概述

- **数据获取与预处理**：支持从 `data` 文件夹中加载多种期货数据，并进行基础清理和格式化。
- **分析与策略实现**：使用 NumPy 和 Pandas 进行数据分析，构建配对交易策略。
- **数据可视化**：通过 Matplotlib 绘制结果图表，帮助理解策略效果。

## 文件结构

```
/data                  # 存放期货数据文件
Future_pair_trading_max.md  # 主代码文件，包含所有逻辑和实现
README.md              # 项目说明文件
```

## 环境要求

请确保安装以下 Python 库：

```bash
pip install numpy pandas matplotlib
```

## 使用方法

1. **克隆项目**：

   ```bash
   git clone https://github.com/your-username/Future_pair_trading.git
   cd Future_pair_trading
   ```

2. **准备数据**：将期货数据文件放置在 `data` 文件夹下。

3. **运行代码**：

   在 Jupyter Notebook 中逐段执行代码。


## 贡献

欢迎提交 Issues 和 Pull Requests 来帮助改进该项目。

## 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 许可。
