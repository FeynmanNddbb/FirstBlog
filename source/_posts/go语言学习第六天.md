---
title: go语言学习第六天
date: 2025-09-08 11:56:14
tags: [Go, 学习, 递归迭代与接口, 编程语言]
cover: /img/golang.jpg
---


## 1.递归与迭代的对比总结

- **递归函数结构**  
  包含基准条件（终止递归）和递归条件（函数自身调用，分解问题）。

- **优点**  
  - 代码简洁，易于理解。
  - 适合分解为相似子问题的问题，如树遍历、分治算法等。

- **缺点**  
  - 性能开销较大，递归调用会占用栈空间，递归深度过大可能导致栈溢出。
  - 调试较困难，递归层次多时不易定位问题。

- **递归与迭代对比**

| 特性       | 递归           | 迭代           |
|------------|----------------|----------------|
| 代码简洁性 | 通常更简洁     | 可能更冗长     |
| 性能       | 可能较慢，占用栈空间 | 通常更快，占用较少内存 |
| 适用场景   | 适合分解为子问题的问题 | 适合线性或简单重复的问题 |

**总结**：递归适合解决可分解为子问题的复杂场景，代码简洁但有性能和调试上的劣势；迭代适合处理线性、重复性强的问题，性能更优但代码可能冗长。

## 2.数值数据类型转换、字符串数据类型转换√

## 3.接口类型转换有两种情况：类型断言和类型转换？（方法？）
1.定义接口、定义结构体；
2.结构体实现接口的所有方法；
3.结构体赋实例，再赋值给接口变量（数据类型为接口的变量）；
4.接口变量直接调用接口方法。

**接口类型变量可以存储任何实现了该接口的类型的值**
``` go
package main

import (
        "fmt"
		"math"
)

// 1.定义接口
type Shape interface {
        Area() float64
        Perimeter() float64
}

// 2.定义结构体 Rectangle
type Rectangle struct {
        Long float64
		Short float64
}
// 定义结构体 Rectangle
type Circle struct {
        Radius float64
}

// 3.Rectangle 实现 Shape 接口
func (c Rectangle) Area() float64 {
        return c.Long * c.Short
}

func (c Rectangle) Perimeter() float64 {
        return 2 * (c.Long + c.Short)
}
// Circle 实现 Shape 接口
func (c Circle) Area() float64 {
        return math.Pi * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
        return 2 * math.Pi * c.Radius
}
func main() {
	var s Shape//接口类型变量可以存储任何实现了该接口的类型的值
    4.s= Rectangle{Long: 5,Short:3}//实现了该接口的类型的值
	fmt.Println("圆形：")
    fmt.Println("面积为：", s.Area())
    fmt.Println("周长为：", s.Perimeter())
	s= Circle{Radius: 5}//实现了该接口的类型的值
	fmt.Println("矩形：")
    fmt.Println("面积为：", s.Area())
    fmt.Println("周长为：", s.Perimeter())
}
```

## 4.next:错误处理