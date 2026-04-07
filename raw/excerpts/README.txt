多篇文章可放在 book_excerpt.txt，篇首特征行见 text_sources.py 中的切分规则。
分文件可放在 raw/excerpts/articles/*.txt（有该目录下 txt 时不再读 book_excerpt.txt）。

NER/实体链接参数：sources/ner_link_config.json
构建与导出唯一入口：在项目根目录执行  python run.py
