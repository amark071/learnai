---
layout: page
title: Links
description: 没有链接的博客是孤独的
keywords: 知识链接
comments: true
menu: 链接
permalink: /links/
---

> 他山之石，可以攻玉

<ul>
{% for link in site.data.links %}
  {% if link.src == 'life' %}
  <li><a href="{{ link.url }}" target="_blank">{{ link.name}}</a></li>
  {% endif %}
{% endfor %}
</ul>
