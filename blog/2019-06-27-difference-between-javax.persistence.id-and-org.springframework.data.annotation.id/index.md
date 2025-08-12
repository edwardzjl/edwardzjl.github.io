---
slug: difference-between-javax.persistence.id-and-org.springframework.data.annotation.id
authors: [jlzhou]
tags: [spring, java]
---

# [译] javax.persistence.Id 和 org.springframework.data.annotation.Id 的区别

## org.springframework.data.annotation.Id

`org.springframework.data.annotation.Id` 是 Spring 定义的 annotation，用来支持 "没有像 JPA 那样的持久化 API" 的非关系型数据库或是框架的持久化，因此它常被用于其它 spring-data 项目，例如 spring-data-mongodb 和 spring-data-solr 等。

## javax.persistence.Id

`javax.persistence.Id` 是由 JPA 定义的 annotation，JPA 仅适用于关系数据的管理。

<!-- truncate -->

## Ref

- [whats-the-difference-between-javax-persistence-id-and-org-springframework-data](https://stackoverflow.com/questions/39643960/whats-the-difference-between-javax-persistence-id-and-org-springframework-data)
