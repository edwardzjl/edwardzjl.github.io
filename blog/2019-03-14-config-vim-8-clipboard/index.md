---
slug: config-vim-8-clipboard
title: vim 8.0 剪切板设置
authors: [jlzhou]
tags: [vim]
---

通过 `ubuntu` 和 `centos` 的源安装的 `vim` 版本较老（好像是7.4.x）

8.0 之后的 `vim`，官网推荐的安装方式是从 git clone 源码编译

默认编译出来的 `vim` 是没有 clipboard support 的，无法通过寄存器与系统剪切板进行交互

在编译时增加 clip board support 需要的最小依赖为 `xorg header files` 和 `x11 dbus`

在 <https://packages.ubuntu.com> 里一通搜索发现 `xorg header files` 是在 `libx11-dev` 这个包里，而 `x11 dbus` 在 `dbus-x11`

因此整个编译过程如下：

```sh
sudo apt-get install libx11-dev dbus-x11
./configure --with-features=huge
make
sudo make install
```
