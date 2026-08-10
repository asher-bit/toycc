# 附录 A：读编译器源码的最小 C++ 手册（从基础 C 起步）

> 目标：**不是学会写 C++**，而是"够看懂 TVM / LLVM / MLIR 的源码和 PR"。
> 你已有基础 C，本手册只讲 C → C++ 的过渡 + 编译器代码里的常见模式。
> 每节配一个"真实源码长这样"的例子。
>
> 读完你应能做到：打开任意一个 `.cc` 文件，能看出"它在定义什么、
> 传什么参数、返回什么、遍历什么"。

---

## 0. 心态：把 C++ 当"带东西的 C"来读

C 你已经会：变量、函数、`struct`、指针、`for` 循环。
C++ = C + 以下几样**新增的东西**。你只需要认得它们：

```
C:    struct Point { int x, y; };
C++:  class Tensor { public: ... };    ← 结构体里能放函数
C:    int *p = &a;                      ← 指针
C++:  int &r = a;                       ← 引用(自动解引用的指针, 更安全)
C:    int add(int a, int b);
C++:  std::vector<int> v;               ← 自带大小的数组
C++:  template <typename T>             ← "任意类型"的泛型写法
```

**读源码时你的心理翻译器**：遇到看不懂的 C++ 语法，就把它翻译成
Python 心理模型。

---

## 1. C → C++ 的五个最小跳跃（必须吃透）

### 1.1 引用 `&`——"安全指针"

```cpp
void foo(int &x) {        // x 是引用: 传进来的是"同一个东西"
    x = 42;               // 直接改原变量
}
int a = 0;
foo(a);                   // a 变成 42
```

**翻译成 Python**：引用 ≈ Python 里"传的是对象本身"（可修改）。
区别只是 C++ 里你要在参数类型后写 `&`。

**为什么编译器代码满屏 `&`？** 因为传大对象（Tensor、Graph）时，
用引用避免拷贝（拷贝 = 复制整份数据，很贵）。**看到 `const T &x` =
"传引用但不让改" = 只读引用**。这个最常出现：

```cpp
bool IsValid(const Tensor &t) {   // 读 t, 但不改它
    return t.size() > 0;
}
```

### 1.2 `class` + `public/private`——"能放函数的 struct"

```cpp
class Tensor {
public:                    // 这下面的都能被外面用
    int size() { return n_; }
private:                   // 这下面的只有内部能用
    int n_ = 0;
};
```

**翻译成 Python**：就是类，`public` = 公开方法，`private` = 下划线开头。

### 1.3 STL 容器——"自带方法的数组和字典"

编译器代码最常见的三个：

```cpp
std::vector<int> v;          // 动态数组 (Python list)
v.push_back(3);              // .append(3)
v.size();                    // len(v)
v[i];                        // v[i]

std::map<std::string, int> m;  // 字典 (Python dict, 但有序)
m["a"] = 1;                    // m["a"] = 1
if (m.count("a")) {...}        // "a" in m

std::string s = "hello";       // 字符串
s + " world";                  // 拼接
```

**翻译**：`std::vector` = list，`std::map` = dict，`std::string` = str。
看到 `std::` 前缀 = "标准库的"。

### 1.4 `auto` 和 range-for——"让编译器猜类型 + 遍历"

```cpp
auto x = GetTensor();      // auto = 让编译器推断类型 (Python 不需要类型名)
for (auto &n : nodes) {    // 遍历 nodes 的每一个 (Python for n in nodes)
    ...n...
}
```

`auto` 让你不用写冗长的类型名；range-for 让你遍历容器。
**这两个是源码里最常用的**，一定要认。

### 1.5 `nullptr` 和 `const`

```cpp
Tensor *t = nullptr;      // NULL, 但类型安全
const int N = 8;          // 常量 (只读)
```

---

## 2. 模板 `template`——只要会"看"就够

模板 = 泛型：同一个函数/类，能作用于任意类型。

```cpp
template <typename T>
T max_of(T a, T b) {        // T 是占位符, 使用时才知道具体类型
    return a > b ? a : b;
}
int x = max_of(3, 5);            // T = int
double y = max_of(3.5, 2.0);     // T = double
```

**你只需要看懂三点**：
1. `template <typename T>` 开头 = 这是个泛型
2. `std::vector<int>` 里的 `<int>` = "这个容器装 int"
3. 看到 `<...>` 就当作 Python 的类型标注

**编译器源码里的模板**（TVM 大量使用）：
```cpp
auto info = GetAttrs<TirOpPattern>();    // 拿一个"TirOpPattern 类型的属性"
```

---

## 3. 智能指针——编译器对象的"自动回收"方式

C 里你用 `malloc/free` 手动管理内存，容易漏（内存泄漏）或重复释放（崩溃）。
C++ 用**智能指针**自动管理：

```cpp
std::shared_ptr<Tensor> p = std::make_shared<Tensor>();  // 共享所有权, 引用计数
std::unique_ptr<Tensor> q = std::make_unique<Tensor>();  // 独占所有权
// 不用手动 delete! 作用域结束自动释放
```

**读源码要点**：
- `shared_ptr` = "多个地方能引用它"（Python 的对象引用）
- `unique_ptr` = "只有一个拥有者"
- 编译器 IR 对象大量用这类指针——看到 `Ptr`、`Ref` 后缀基本是这个意思

---

## 4. 编译器代码的"惯用法"（最重要的一节）

下面这些是 TVM/LLVM/MLIR 里**天天出现**的模式，认得它们，
源码就通了 80%。

### 4.1 模式一：`T::GetRef<T>()` / `as<T>()` / `down_cast<T>()`——"安全地当某个类型用"

编译器的 IR 是"一个基类 + 很多子类"：

```
Expr(基类: 所有表达式的公共父类)
 ├── Var       (变量)
 ├── Constant  (常量)
 ├── Call      (函数调用)
 └── ...更多
```

拿到一个 `Expr`，你想知道它具体是哪种？用 `as<T>()` 试转：

```cpp
Expr e = GetSomeExpr();
if (auto call = e.as<Call>()) {     // 如果 e 是 Call, 转成功(非空)
    // 这里可以把 call 当 Call 用
    int num_args = call->args.size();
}
```

**翻译成 Python**：≈ `isinstance(e, Call)` + 转型。
`as<Call>()` 失败返回空指针 → 用 `if (auto ...)` 判断。

**`down_cast<T>(x)`**：确定 x 一定是 T 时才用（省掉检查，更快）。

### 4.2 模式二：`ObjectRef` 体系——"值语义但共享底层"

TVM 的对象是这样设计的：

```cpp
// 你看到一个 "Ref" 结尾的类型: 它是"共享的对象句柄"
Tensor x = ...;
// 赋值只是复制句柄, 底层数据共享 (像 Python 的引用)
Tensor y = x;
```

- `ObjectRef` = 句柄（类似 Python 引用）
- 底层 `Object` 是共享数据
- 好处：传参拷贝便宜（只是复制句柄），语义像"值"但内存共享

**读源码提示**：看到 `Expr`、`Var`、`Call` 这类类型，它们都是一种
"引用"，赋值不深拷贝。这就是第 1 课深拓展讲的"不可变 + 结构共享"。

### 4.3 模式三：`ExprVisitor` / `ExprMutator`——"遍历/改写 IR"

编译器要遍历整棵 IR 树。TVM 封装成两个基类：

```cpp
// Visitor: 只读遍历
class MyVisitor : public ExprVisitor {
public:
    void VisitExpr_(const CallNode *op) override {
        // 遇到每个 Call 节点就执行这里
        LOG(INFO) << "call with " << op->args.size() << " args";
        ExprVisitor::VisitExpr_(op);   // 继续遍历子节点
    }
};

// Mutator: 遍历并改写
class MyMutator : public ExprMutator {
public:
    Expr VisitExpr_(const CallNode *op) override {
        // 可以返回一个新的节点来替换它
        return SomeNewExpr(...);
    }
};
```

**翻译成 Python**：这就是"访问者模式"——给每类节点一个回调。
看到 `VisitExpr_(const XxxNode*)` 就知道"对每种 Xxx 节点做什么"。

### 4.4 模式四：`static const Op& op = Op::Get("relax.call_tir")`——"查注册表"

前面课程讲过算子注册表。源码里拿一个算子：

```cpp
static const Op &call_tir_op = Op::Get("relax.call_tir");
if (call->op.same_as(call_tir_op)) {   // 判断"这是不是 call_tir"
    ...
}
```

`Op::Get("名字")` = 按名字查注册表（第 1 课 `OPS[name]`）。

### 4.5 模式五：宏——`TVM_FFI_ICHECK` / `LOG` / `TVM_REGISTER_GLOBAL`

宏 = 编译前替换的代码。看到大写函数名一般是宏：

```cpp
TVM_FFI_ICHECK(x > 0) << "x must be positive";   // 条件不满足就报错退出 (= assert)
LOG(INFO) << "start fusing";                     // 打日志 (= print)
TVM_REGISTER_GLOBAL("relax.FuseOps")             // 注册进 FFI 全局表
    .set_body_typed(FuseOps);                    //   绑定到 C++ 函数
```

**翻译**：
- `TVM_FFI_ICHECK(cond)` = `assert(cond)`
- `LOG(INFO) << x` = `print(x)`
- `TVM_REGISTER_GLOBAL(...)` = 注册表登记（第 8 课 FFI）

---

## 5. 一个"读真实源码"的实战演练

我们拿第 8 课见过的 `fold_constant.cc` 里一段，用本手册逐行读：

```cpp
bool ShouldBeFolded(Expr expr) {                    // 函数: 入参 expr(只读引用省略了&)
  static constexpr int64_t kMaxFoldElements = 1024; // 常量 1024
  ...
  if (num_elements <= kMaxFoldElements) return true;// 小就返回 true
  for (const auto &arg : call->args) {              // 遍历 call 的所有参数
    if (ExprContainsTensor(arg)) return true;       // 参数里有张量就 true
  }
  return false;
}
```

**逐行翻译成你的 Python 直觉**：

```python
def should_be_folded(expr):
    kMax = 1024
    if num_elements(expr) <= kMax:
        return True
    for arg in call.args:            # range-for
        if expr_contains_tensor(arg):
            return True
    return False
```

看到了吗？**你完全读得懂**。C++ 只是语法外壳，逻辑和 Python 一样。

再看 `ExprMutator` 用法：

```cpp
Expr VisitExpr_(const CallNode *call) final {   // 对每个 Call 节点
  if (!ShouldBeFolded(post_call)) return post_call;   // 不折就原样返回
  ...
  return VisitCallTIR(post_call).value_or(post_call); // 折了或原样
}
```

**翻译**：`VisitExpr_(CallNode)` = "处理 Call 类型节点的回调"；
`value_or(default)` = "有值就取，没有就用 default"（≈ Python 的 `or`）。

---

## 6. 读源码的通用翻译技巧

1. **看到类型名看不懂** → 想"这是个啥对象"，看它在干嘛，别查类型定义
2. **看到 `->`**（如 `op->args`）→ "指针的成员"（C 里你见过）
3. **看到 `.`**（如 `expr.as<Call>()`）→ "对象的成员方法"
4. **看到 `auto`** → 忽略类型，看右边表达式
5. **看到 `const T &x`** → "只读地传 x，不拷贝"
6. **看到 `std::vector/string/map`** → list/str/dict
7. **看到 `if (auto x = f())`** → "f() 成功才有 x"（空指针即失败）
8. **看到 `XXNode`** → 底层节点类；`XX`（不带 Node）→ 它的引用/句柄
9. **看到大写宏** → 当 assert / print / 注册 理解
10. **卡住超过 1 分钟** → 跳过，先抓主逻辑（for 循环里在干嘛）

**最重要的心态**：读懂"逻辑"（遍历什么、判断什么、返回什么）就够了，
不需要读懂每个 C++ 语法细节。你的目标是**参与讨论、看懂 PR 的意图**，
不是成为 C++ 专家。

---

## 7. 本附录小结

- C++ = C + 引用/类/STL/template/智能指针
- 编译器惯用法五件套：`as<>()` 试转、`ObjectRef` 句柄、
  `ExprVisitor/Mutator` 遍历改写、`Op::Get` 查注册表、`TVM_*` 宏
- 读源码 = 用"心理翻译器"转成 Python 逻辑
- 十个翻译技巧 + 一个心态：**看懂逻辑就够**

**下一步**：读第 16 课（真实工程开发流程）。但在这之前，你需要一个
能跑 TVM 的环境——看附录 B（Windows → WSL2 环境搭建）。
