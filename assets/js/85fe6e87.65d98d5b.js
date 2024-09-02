"use strict";(self.webpackChunkedwardzjl_github_io=self.webpackChunkedwardzjl_github_io||[]).push([[681],{5680:(e,t,r)=>{r.d(t,{xA:()=>d,yg:()=>u});var n=r(6540);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function s(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},o=Object.keys(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var l=n.createContext({}),p=function(e){var t=n.useContext(l),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},d=function(e){var t=p(e.components);return n.createElement(l.Provider,{value:t},e.children)},c="mdxType",g={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},f=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,o=e.originalType,l=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),c=p(r),f=a,u=c["".concat(l,".").concat(f)]||c[f]||g[f]||o;return r?n.createElement(u,i(i({ref:t},d),{},{components:r})):n.createElement(u,i({ref:t},d))}));function u(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=r.length,i=new Array(o);i[0]=f;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s[c]="string"==typeof e?e:a,i[1]=s;for(var p=2;p<o;p++)i[p]=r[p];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}f.displayName="MDXCreateElement"},8203:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>l,contentTitle:()=>i,default:()=>g,frontMatter:()=>o,metadata:()=>s,toc:()=>p});var n=r(8168),a=(r(6540),r(5680));const o={slug:"difference-between-javax.persistence.id-and-org.springframework.data.annotation.id",title:"<\u8bd1> javax.persistence.Id \u548c org.springframework.data.annotation.Id \u7684\u533a\u522b",authors:["jlzhou"],tags:["spring","java"]},i=void 0,s={permalink:"/blog/difference-between-javax.persistence.id-and-org.springframework.data.annotation.id",editUrl:"https://github.com/edwardzjl/edwardzjl.github.io/blob/main/blog/2019-06-27-difference-between-javax.persistence.id-and-org.springframework.data.annotation.id/index.md",source:"@site/blog/2019-06-27-difference-between-javax.persistence.id-and-org.springframework.data.annotation.id/index.md",title:"<\u8bd1> javax.persistence.Id \u548c org.springframework.data.annotation.Id \u7684\u533a\u522b",description:"org.springframework.data.annotation.Id",date:"2019-06-27T00:00:00.000Z",formattedDate:"2019\u5e746\u670827\u65e5",tags:[{label:"spring",permalink:"/blog/tags/spring"},{label:"java",permalink:"/blog/tags/java"}],readingTime:.405,hasTruncateMarker:!1,authors:[{name:"Junlin Zhou",title:"Fullstack Engineer @ ZJU ICI",url:"https://github.com/edwardzjl",imageURL:"https://github.com/edwardzjl.png",key:"jlzhou"}],frontMatter:{slug:"difference-between-javax.persistence.id-and-org.springframework.data.annotation.id",title:"<\u8bd1> javax.persistence.Id \u548c org.springframework.data.annotation.Id \u7684\u533a\u522b",authors:["jlzhou"],tags:["spring","java"]},prevItem:{title:"\u7cfb\u7edf\u4e2d\u72b6\u6001\u4e3a static \u7684\u670d\u52a1",permalink:"/blog/static-service"},nextItem:{title:"Install postgres on OSX",permalink:"/blog/install-postgres-on-osx"}},l={authorsImageUrls:[void 0]},p=[{value:"org.springframework.data.annotation.Id",id:"orgspringframeworkdataannotationid",level:2},{value:"javax.persistence.Id",id:"javaxpersistenceid",level:2},{value:"Ref",id:"ref",level:2}],d={toc:p},c="wrapper";function g(e){let{components:t,...r}=e;return(0,a.yg)(c,(0,n.A)({},d,r,{components:t,mdxType:"MDXLayout"}),(0,a.yg)("h2",{id:"orgspringframeworkdataannotationid"},"org.springframework.data.annotation.Id"),(0,a.yg)("p",null,(0,a.yg)("inlineCode",{parentName:"p"},"org.springframework.data.annotation.Id"),' \u662f Spring \u5b9a\u4e49\u7684 annotation\uff0c\u7528\u6765\u652f\u6301 "\u6ca1\u6709\u50cf JPA \u90a3\u6837\u7684\u6301\u4e45\u5316 API" \u7684\u975e\u5173\u7cfb\u578b\u6570\u636e\u5e93\u6216\u662f\u6846\u67b6\u7684\u6301\u4e45\u5316\uff0c\u56e0\u6b64\u5b83\u5e38\u88ab\u7528\u4e8e\u5176\u5b83 spring-data \u9879\u76ee\uff0c\u4f8b\u5982 spring-data-mongodb \u548c spring-data-solr \u7b49\u3002'),(0,a.yg)("h2",{id:"javaxpersistenceid"},"javax.persistence.Id"),(0,a.yg)("p",null,(0,a.yg)("inlineCode",{parentName:"p"},"javax.persistence.Id")," \u662f\u7531 JPA \u5b9a\u4e49\u7684 annotation\uff0cJPA \u4ec5\u9002\u7528\u4e8e\u5173\u7cfb\u6570\u636e\u7684\u7ba1\u7406\u3002"),(0,a.yg)("h2",{id:"ref"},"Ref"),(0,a.yg)("ul",null,(0,a.yg)("li",{parentName:"ul"},(0,a.yg)("a",{parentName:"li",href:"https://stackoverflow.com/questions/39643960/whats-the-difference-between-javax-persistence-id-and-org-springframework-data"},"whats-the-difference-between-javax-persistence-id-and-org-springframework-data"))))}g.isMDXComponent=!0}}]);