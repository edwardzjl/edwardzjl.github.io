---
slug: config-vim-8-clipboard
authors: [jlzhou]
tags: [vim]
---

# Ubuntu / CentOS 编译安装带剪贴板支持的 Vim

通过 Ubuntu 或 CentOS 系统自带的软件源安装 Vim，往往只能得到较旧的版本（通常是 7.4.x）。而从 Vim 8.0 开始，官网推荐的安装方式是通过 Git 克隆源码自行编译。

不过需要注意，**默认编译出来的 Vim 并不包含剪贴板支持（clipboard support）**，因此无法与系统剪贴板交互（例如复制粘贴到其他程序）。

<!-- truncate -->

要在编译时启用剪贴板支持，至少需要两个依赖包：

* `libx11-dev`：提供 Xorg 的头文件（xorg header files）
* `dbus-x11`：提供 X11 的 D-Bus 支持

你可以在 [https://packages.ubuntu.com](https://packages.ubuntu.com) 搜索具体的依赖项位置，最终确认这两个包就是我们所需的。

安装依赖并编译 Vim 的完整流程如下：

```sh
sudo apt-get install libx11-dev dbus-x11
git clone https://github.com/vim/vim.git
cd vim
./configure --with-features=huge --enable-gui=auto --enable-cscope --prefix=/usr/local
make
sudo make install
```

> 其中 `--with-features=huge` 启用几乎所有功能，`--enable-gui=auto` 可选启用 GUI 模式（如 gvim），`--enable-cscope` 则用于增强代码导航功能。

安装完成后，你可以使用 `vim --version` 检查是否启用了 `+clipboard`，确认剪贴板支持是否生效。
