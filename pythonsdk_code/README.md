# Python 控制 Knowm Memristor Discovery (AD3) — 测试阶梯

环境：`C:\Users\10401\.workbuddy\binaries\python\envs\default\Scripts\python.exe`
（已装 pydwf + numpy + matplotlib；dwf.dll 来自 D:\Digilent\WaveForms3，v3.25.1）

## 运行顺序（每步通过再进下一步）

| # | 脚本 | 测什么 | 需要的接线 | 通过判据 |
|---|------|--------|-----------|---------|
| 0 | `00_connect_test.py` | Python 能枚举到 AD3 | 插上 USB 即可 | "PYDWF LINK OK" |
| 1 | `01_loopback.py` | AWG→Scope 全链路 | 跳线：W1→1+，GND→1- | 幅度≈1V，频率≈1kHz，存 loopback_result.png |
| 2 | `02_power_and_dio.py` | ±5V 电源轨 + MUX 控制线 | 板子插在 AD3 上 | V+≈+5V，V-≈-5V，DIO 回读高 8 位=00011101 |

## 通过后接下去做什么

3. **读单个忆阻器电导（FLV 的 Python 版）**：固定 W1 输出小幅度正弦/DC，
   Scope CH1 测忆阻器电压、CH2 测串联电阻电压 → G = I/V。
4. **写单个忆阻器（SET/RESET）**：AWG1 输出单脉冲（正=SET，负=RESET），脉宽 µs 级。
5. **8 突触 AHAH 训练循环**：复现 GUI 的 Classify21，数据集/规则可自由改。

## 硬件事实备忘（从 knowm/memristor-discovery 源码核实）

- 电源：V+ = +5.0 V，V- = −5.0 V（DWFProxy.setPowerSupply）
- DIO 高 8 位（DIO8–15）控制板上 4 个 MUX 的路由，顺序（MSB→LSB）：
  W2(2bit) W1(2bit) 2+(2bit) 1+(2bit)；每 2bit：00=OUT 10=Y 01=A 11=B
- GUI 默认路由：W1→A、W2→OUT、scope1→A、scope2→B
  = `0b00011101_00000000`（即 0x1D00）
- DIO 低 8 位：memristor 行选地址（具体位序待上电标定）
- AWG 通道 0 = W1，Scope 通道 0 = CH1（GUI 源码用 WAVEFORM_CHANNEL_1 / OSCILLOSCOPE_CHANNEL_1）
