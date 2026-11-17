---
title: Python 从入门到精通（一）：基础认知与核心语法
date: 2026-10-03 10:00:00
tags: [Python,编程语言,基础]
categories: [教程]
description: 从 Python 的基本概念出发，系统认识变量、标识符、关键字、数据类型、输入输出和运算符，建立后续学习 Python、PyTorch 与大模型工程的基础。
---

# Python 从入门到精通（一）：基础认知与核心语法

这是“Python 从入门到精通”专题的第一个模块。本模块先建立一张清晰的知识地图：Python 程序由什么组成，变量究竟保存了什么，数据有哪些类型，以及如何使用运算符处理数据。

后面学习函数、面向对象、文件处理、Web 开发、PyTorch 或大模型部署时，都会反复用到这些基础概念。

## 一、Python 到底是什么？

Python 是一种**高级、通用、动态类型**的编程语言。理解这句话，需要先认识几个术语：

| 术语 | 英文 | 通俗解释 |
| --- | --- | --- |
| 编程语言 | Programming Language | 人类编写程序时使用的语言 |
| 高级语言 | High-level Language | 更接近人类表达，隐藏了许多底层细节 |
| 解释型语言 | Interpreted Language | 通常通过解释器参与执行程序 |
| 动态类型 | Dynamic Typing | 变量在运行时可以绑定不同类型的对象 |
| 解释器 | Interpreter | 负责执行 Python 程序的软件 |
| 语法 | Syntax | 代码必须遵守的书写规则 |
| 运行时 | Runtime | 程序实际执行的环境 |

Python 源代码通常会先被编译成字节码，再由 Python 虚拟机执行。因此，“解释型语言”是便于理解的概括，并不意味着每一行代码都简单地被逐字翻译执行。

最简单的 Python 程序是：

```python
print("Hello Python")
```

这段代码中：

- `print` 是 Python 内置的输出函数；
- `()` 表示调用函数；
- `"Hello Python"` 是字符串；
- `print()` 会把内容输出到控制台。

运行结果：

```text
Hello Python
```

### Python 和 C++ 有什么区别？

| 对比项目 | Python | C++ |
| --- | --- | --- |
| 类型系统 | 动态类型 | 静态类型为主 |
| 内存管理 | 由运行时自动管理，并带有垃圾回收机制 | 手动管理与自动管理机制并存 |
| 语法 | 相对简洁 | 功能强大，语法相对复杂 |
| 执行方式 | 通常执行字节码 | 通常编译为机器码 |
| 典型用途 | AI 应用、Web、自动化、脚本 | 高性能计算、游戏、底层系统 |

这里要建立一个重要认知：**使用 Python，不等于所有计算都由 Python 解释器低效完成。**

例如在 PyTorch 中：

```python
import torch

a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")

c = torch.matmul(a, b)
```

你写的是 Python，但矩阵乘法通常会交给底层 C++、CUDA 和 GPU Kernel 执行。Python 更像是负责组织流程和调用能力的上层语言，这也是后面学习 PyTorch、vLLM 和算子开发的基础。

## 二、变量：名称如何引用对象

变量可以理解为程序中用于引用某个对象的名称：

```python
name = "Qwen"
age = 27
gpu = "RTX 4090"
```

这里：

- `name`、`age`、`gpu` 是变量名；
- `"Qwen"`、`27`、`"RTX 4090"` 是对象；
- `=` 是赋值运算符。

注意，Python 中的 `=` 不是数学意义上的“相等”，而是**让变量名绑定到某个对象**。

### 变量的本质：名称绑定对象

```python
a = 10
b = a

print(a)
print(b)
```

输出：

```text
10
10
```

执行 `b = a` 后，`a` 和 `b` 都引用整数对象 `10`。如果之后执行：

```python
a = 20

print(a)  # 20
print(b)  # 10
```

`b` 不会跟着变成 `20`。因为 `a = 20` 做的是重新绑定：让 `a` 指向对象 `20`，并没有修改原来的整数对象。

这个“变量是名称、对象才是数据”的认识，是理解 Python 引用关系、可变对象和内存行为的基础。

## 三、标识符：给变量和程序元素命名

标识符是程序中给变量、函数、类等元素起的名字：

```python
user_name = "Tom"
model_size = 27
```

### 命名规则

合法命名：

```python
name = "Tom"
_age = 20
model2 = "Qwen"
```

非法命名：

```python
2name = "Tom"      # 不能以数字开头
user-name = 1       # 连字符会被解释为减号
class = 10          # class 是关键字
```

常见命名风格：

| 风格 | 示例 | 常见用途 |
| --- | --- | --- |
| `snake_case` | `model_name` | 变量、函数 |
| `PascalCase` | `ModelRunner` | 类名 |
| `UPPER_CASE` | `MAX_LENGTH` | 常量约定 |

命名时应优先表达含义。相比 `x`、`data1`，`model_name`、`max_length` 更容易阅读和维护。

## 四、关键字：Python 预留的特殊单词

关键字是 Python 语言预留的单词，具有固定的语法含义，不能直接用作变量名：

```text
if       else      for       while
def      class     return    import
from     True      False     None
and      or        not
```

例如：

```python
age = 25

if age > 18:
    print("成年")
```

其中：

- `if` 是条件判断关键字；
- `age > 18` 是条件表达式；
- `:` 表示后面开始一个代码块；
- 缩进表示代码块的范围。

Python 使用缩进表达层级。通常使用 **4 个空格**表示一级缩进，不建议在同一个文件中混用 Tab 和空格。

## 五、注释和文档字符串

注释是给程序员看的说明文字，普通注释不会作为程序语句执行：

```python
# 这是一个变量
name = "Qwen"
```

多行说明可以使用多行字符串：

```python
"""
这是一个程序说明。
可以写多行内容。
"""
```

函数、类和模块中的说明通常使用**文档字符串**：

```python
def add(a, b):
    """计算两个数字的和。"""
    return a + b
```

文档字符串会被 Python 保存下来，可以通过 `help()` 或 `函数.__doc__` 查看；它和普通注释的用途并不完全相同。

## 六、Python 的输入与输出

### 1. 输出：`print()`

```python
print("Hello")
print(100)
print(3.14)
```

`print()` 也可以一次输出多个值：

```python
name = "Qwen"
size = 27

print(name, size)
```

如果需要把变量嵌入一段文字，可以使用 f-string：

```python
name = "Qwen"
size = 27

print(f"模型名称：{name}，参数量：{size}B")
```

f-string 是格式化字符串，花括号中的表达式会在运行时求值。

### 2. 输入：`input()`

```python
name = input("请输入你的名字：")

print(f"你好，{name}")
```

`input()` 返回值默认是字符串 `str`。即使输入 `25`，得到的也是字符串 `"25"`：

```python
age = input("请输入年龄：")

print(type(age))  # <class 'str'>
```

如果需要整数，必须显式转换：

```python
age = int(input("请输入年龄："))

print(age + 1)
```

如果输入的内容不能转换成整数，例如 `hello`，`int()` 会抛出 `ValueError`。后续学习异常处理后，可以让程序更友好地处理这种情况。

## 七、Python 的基础数据类型

### 1. 什么是数据类型？

数据类型（Data Type）用于说明一个对象属于哪一类，以及它支持哪些操作：

```python
a = 100
b = 3.14
c = "Hello"
d = True
```

这里的对象分别属于整数、浮点数、字符串和布尔类型。

### 2. 八类常用基础类型

| 类型 | 英文 | 示例 | 用途 |
| --- | --- | --- | --- |
| 整数 | `int` | `10` | 整数计算 |
| 浮点数 | `float` | `3.14` | 小数计算 |
| 布尔值 | `bool` | `True` | 条件判断 |
| 字符串 | `str` | `"Python"` | 文本处理 |
| 列表 | `list` | `[1, 2, 3]` | 有序、可变集合 |
| 元组 | `tuple` | `(1, 2, 3)` | 有序、不可变集合 |
| 字典 | `dict` | `{"a": 1}` | 键值映射 |
| 集合 | `set` | `{1, 2, 3}` | 不重复元素集合 |

此外还有一个非常重要的特殊值：

```python
result = None
```

`None` 表示空值、缺少结果或“目前没有具体对象”，它的类型是 `NoneType`。

### 3. 使用 `type()` 查看类型

```python
a = 100
b = 3.14
c = "Python"
d = True
e = None

print(type(a))
print(type(b))
print(type(c))
print(type(d))
print(type(e))
```

输出：

```text
<class 'int'>
<class 'float'>
<class 'str'>
<class 'bool'>
<class 'NoneType'>
```

这里的 `class` 表示类型本身也是一个类。面向对象章节会进一步解释类、对象和实例之间的关系。

### 4. 类型转换

Python 不会自动把所有类型随意转换成另一种类型。需要时，可以调用对应的类型构造函数：

```python
a = "100"
b = int(a)

print(b + 10)  # 110
```

常见转换方式：

```python
int("100")        # 字符串转整数
float("3.14")     # 字符串转浮点数
str(100)           # 数字转字符串
bool(1)            # 转布尔值
list((1, 2, 3))    # 元组转列表
tuple([1, 2, 3])   # 列表转元组
set([1, 1, 2])     # 列表转集合，结果为 {1, 2}
```

转换失败时会抛出异常：

```python
int("hello")
```

上面的代码会产生 `ValueError`，因为字符串 `"hello"` 不是合法的整数表示。

## 八、运算符：对数据执行操作

运算符是对数据执行计算、比较或逻辑判断的符号。

### 1. 算术运算符

```python
a = 10
b = 3

print(a + b)   # 13
print(a - b)   # 7
print(a * b)   # 30
print(a / b)   # 3.333...
print(a // b)  # 3
print(a % b)   # 1
print(a ** b)  # 1000
```

几个容易混淆的运算符：

| 运算符 | 名称 | 含义 |
| --- | --- | --- |
| `/` | 除法 | 返回普通除法结果，通常是浮点数 |
| `//` | 整除 | 返回向下取整后的商 |
| `%` | 取模 | 返回余数 |
| `**` | 幂运算 | 计算乘方 |

### 2. 比较运算符

```python
a = 10
b = 20

print(a == b)  # False
print(a != b)  # True
print(a > b)   # False
print(a < b)   # True
print(a >= b)  # False
print(a <= b)  # True
```

一定要区分：

- `=` 是赋值；
- `==` 是比较两个对象的值是否相等。

### 3. 逻辑运算符

```python
age = 25
salary = 10000

print(age > 18 and salary > 5000)   # True
print(age > 18 or salary > 20000)   # True
print(not age > 18)                 # False
```

| 运算符 | 含义 |
| --- | --- |
| `and` | 与：两边都为真才为真 |
| `or` | 或：至少一边为真 |
| `not` | 非：对结果取反 |

### 4. 成员运算符

`in` 和 `not in` 用于判断元素是否存在于某个容器中：

```python
names = ["Tom", "Jack", "Alice"]

print("Tom" in names)      # True
print("Bob" not in names)  # True
```

### 5. 身份运算符

身份运算符判断两个变量是否引用同一个对象：

```python
a = [1, 2, 3]
b = a
c = [1, 2, 3]

print(a == c)  # True，值相等
print(a is c)  # False，不是同一个对象
print(a is b)  # True，引用同一个对象
```

因此：

- `==` 比较值是否相等；
- `is` 比较是否为同一个对象。

判断一个值是否为 `None` 时，推荐使用 `is None`：

```python
result = None

if result is None:
    print("暂时没有结果")
```

## 九、这一阶段必须建立的知识框架

可以用下面这条链路理解 Python 程序：

| 概念 | 要回答的问题 |
| --- | --- |
| Python 程序 | 要让计算机完成什么任务？ |
| 语法 Syntax | 代码应该如何书写？ |
| 变量 Variable | 如何给对象起名字并引用它？ |
| 数据类型 Type | 对象是什么数据？支持哪些操作？ |
| 运算符 Operator | 如何计算、比较和判断数据？ |
| 解释器 | 如何执行程序并产生运行结果？ |

把这些概念串起来，就是：**使用符合语法的代码，通过变量引用不同类型的对象，再用运算符对对象进行操作，最后由解释器执行并得到结果。**

## 十、第一阶段练习

不要只阅读示例，建议把下面的代码和问题实际运行一遍。

### 练习 1：变量重新绑定

Python 中，执行下面的代码后，`b` 的值是多少？

```python
a = 10
b = a
a = 20

print(b)
```

答案是 `10`。`a = 20` 只是让 `a` 重新绑定到整数对象 `20`，`b` 仍然引用原来的整数对象 `10`。

### 练习 2：`input()` 的返回类型

`input()` 默认返回哪种类型？

- `int`
- `str`
- `bool`

答案是 `str`。即使用户输入数字字符，`input()` 也会先返回字符串，需要使用 `int()` 或 `float()` 显式转换。

### 练习 3：判断对象身份

判断两个变量是否引用同一个对象，应该使用哪个运算符？

- `==`
- `is`
- `in`

答案是 `is`。`==` 比较值，`is` 判断对象身份，`in` 判断元素是否属于某个容器。

### 练习 4：合法的变量名

下面哪一个是合法的 Python 变量名？

- `2model`
- `model-name`
- `model_name`

答案是 `model_name`。变量名不能以数字开头，也不能包含连字符；连字符会被解释为减号。

## 小结

本模块的重点不是记住大量语法，而是形成以下几个稳定认知：

1. Python 是高级、通用、动态类型语言，底层重计算通常由 C、CUDA 等组件完成。
2. 变量是名称，对象才是真正的数据；赋值本质上是名称绑定。
3. 标识符必须遵守命名规则，关键字不能拿来当普通变量名。
4. `input()` 返回字符串，类型转换需要显式完成。
5. `=`、`==`、`is`、`in` 的含义不同，不能混用。
6. 语法、变量、类型和运算符共同构成 Python 基础知识框架。

掌握这些内容后，下一模块就可以继续学习条件判断、循环、函数以及更完整的程序控制流程。
