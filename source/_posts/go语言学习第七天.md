---
title: go语言学习第七天
date: 2025-09-09 12:17:04
tags: [Go, 学习, 结构体和并发, 编程语言]
cover: /img/golang.jpg
---

## 1.发现结构体对方法的调用都是用地址，而有了接口之后是直接用变量？
``` go
//结构体对方法的调用
type DivideError struct {
        Dividend int
        Divisor  int
}
func (e *DivideError) Error() string {
        return fmt.Sprintf("cannot divide %d by %d", e.Dividend, e.Divisor)
}
//接口对方法的调用
type Shape interface {
        Area() float64
        Perimeter() float64
}
type Circle struct {
        Radius float64
}
func (c Circle) Area() float64 {
        return math.Pi * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
        return 2 * math.Pi * c.Radius
}
```
## 2.并发以及通道
``` go
ch <- i//通道接收数据,不能加等号！！那是赋值语句
c =<- ch//通道发送数据
```

- Go 语言通过 Goroutine 和 Channel 实现高效并发，简化了传统线程模型。
- **Goroutine**：轻量级线程，使用 `go` 关键字启动。
- **Channel**：用于 Goroutine 之间的数据通信。
- **Select 语句**：可同时等待多个 Channel 操作，提升并发处理能力。

### 常见并发问题及解决方法

- **死锁（Deadlock）**：所有 Goroutine 都在等待数据，程序无法继续。  
  - 解决：合理关闭通道，避免无限等待。
  - 解决：使用 Mutex 或 Channel 进行同步，保证数据安全。

**总结**：Go 并发编程简单高效，需注意死锁和数据竞争等问题，合理使用同步工具可提升程序性能和稳定性。