---
title: go语言学习第五天
date: 2025-09-06 11:55:50
tags: [Go, 学习, 切片、集合和指针, 编程语言]
cover: /img/golang.jpg
---

# 回顾
学了array、slice（其实就是一个容量可变的array）、structure、range遍历各个数组和集合的用法、指针。

# 1.Go切片默认容量扩展机制
在 Go 语言中，切片的容量扩展并不是简单的按需分配，而是结合了**容量翻倍策略**和**内存分配器的大小等级取整**机制。这样设计的目的是提升分配效率、减少内存碎片，并优化整体性能。

- Go 的运行时内存分配器会根据预定义的大小等级分配内存块，而不是每次都精确分配所需字节。
- 当切片容量不足时，append 操作会触发扩容，通常采用容量翻倍策略，但最终容量会向上取整到最近的大小等级。
- 例如，追加元素后所需容量为 5，但实际分配的容量为 6，因为 6 个 int 能刚好匹配 48 字节的分配等级。

这种机制虽然可能导致单次分配时多分配一些内存，但能显著提升切片 append 操作的平均性能，减少频繁分配带来的性能损耗。
当然，容量也可以自定义！！

## 操作过程举例
``` go
package main

import "fmt"

func main() {
   var numbers []int
   printSlice(numbers)

   /* 允许追加空切片 */
   numbers = append(numbers, 0)
   printSlice(numbers)

   /* 向切片添加一个元素 */
   numbers = append(numbers, 1)
   printSlice(numbers)

   /* 同时添加多个元素 */
   numbers = append(numbers, 2,3,4) /* 此时的默认扩容到了6而不是5（2+3） ！！！*/
   printSlice(numbers)

   /* 创建切片 numbers1 是之前切片的两倍容量*/
   numbers1 := make([]int, len(numbers), (cap(numbers))*2)

   /* 拷贝 numbers 的内容到 numbers1 */
   copy(numbers1,numbers)
   printSlice(numbers1)   
}

func printSlice(x []int){
   fmt.Printf("len=%d cap=%d slice=%v\n",len(x),cap(x),x)
}
```
# 2.Map注意事项
Map 是引用类型，如果将一个 Map 传递给一个函数或赋值给另一个变量，它们都指向同一个底层数据结构，因此对 Map 的修改会影响到所有引用它的变量。

# 3.指针的易错点
使用指针所指内容之前，必须先给初始化指针（地址），即有了地址才能找内容，否则为空指针！！
## 总结


## 疑问
channel的定义和用法？