from pathlib import Path
import xml.etree.ElementTree as E
import re,json,subprocess,hashlib,html
import markdown
root=Path('/home/nubot/humanlike_drive/references')
out=root/'.archive/zotero_import_2026-11-04'
ns={'rdf':'http://www.w3.org/1999/02/22-rdf-syntax-ns#','dc':'http://purl.org/dc/elements/1.1/','dcterms':'http://purl.org/dc/terms/','bib':'http://purl.org/net/biblio#','z':'http://www.zotero.org/namespaces/export#','link':'http://purl.org/rss/1.0/modules/link/','foaf':'http://xmlns.com/foaf/0.1/'}
for k,v in ns.items():E.register_namespace(k,v)
def q(s):
 a,b=s.split(':');return '{'+ns[a]+'}'+b
def sub(parent,tag,text=None,resource=None):
 el=E.SubElement(parent,q(tag))
 if text is not None:el.text=str(text)
 if resource is not None:el.set(q('rdf:resource'),resource)
 return el
def node(tag,ident):return E.SubElement(tree,q(tag),{q('rdf:about'):ident})
cat={r['folder']:r for r in json.loads((root/'.archive/library_catalog.json').read_text())}
sources={r['directory']:r for r in json.loads((root/'07_抓持承载与拖拉/sources.json').read_text())['papers']}
tree=E.Element(q('rdf:RDF'))
main=node('z:Collection','urn:steering:collection');sub(main,'dc:title','方向盘操作研究')
manifest=[]
for category in sorted(root.glob('[0-9]*')):
 if not category.is_dir():continue
 cid='urn:steering:category:'+category.name[:2];sub(main,'dcterms:hasPart',resource=cid)
 coll=node('z:Collection',cid);sub(coll,'dc:title',category.name)
 for f in sorted(category.glob('*/README.md')):
  folder=f.parent;rel=str(folder.relative_to(root));txt=f.read_text();title=txt.splitlines()[0].removeprefix('# ')
  row=cat.get(rel,{})
  src=sources.get(folder.name,{}) if category.name.startswith('07') else {}
  ident='urn:steering:paper:'+hashlib.sha256(rel.encode()).hexdigest()[:16]
  paper=node('bib:Document',ident);sub(paper,'z:itemType','document');sub(paper,'dc:title',title)
  sub(coll,'dcterms:hasPart',resource=ident)
  urls=re.findall(r'https?://[^\s)\]>]+',txt)
  url=row.get('arxiv_url') or src.get('id') or row.get('url') or (urls[0] if urls else '')
  if url:sub(paper,'z:url',url)
  sub(paper,'dc:subject','方向盘操作研究');sub(paper,'dc:subject',folder.name)
  if row.get('priority',100)<=10:sub(paper,'dc:subject','核心参考')
  extra='Local reference: '+rel
  if '待核实' in title:sub(paper,'dc:subject','题名待核实')
  m=re.search(r'arxiv.org/(?:abs|pdf)/(\d{4}\.\d+)',url)
  if m:
   sub(paper,'dc:date',src.get('published','')[:10] or f'20{m[1][:2]}-{m[1][2:4]}')
   extra+='\narXiv: '+m[1]
  sub(paper,'dc:description',extra)
  attachments=[]
  for filename,mime in [('paper.pdf','application/pdf'),('paper.html','text/html')]:
   path=folder/filename
   if not path.exists():continue
   aid=ident+':'+filename
   sub(paper,'link:link',resource=aid)
   a=node('z:Attachment',aid);sub(a,'dc:title',folder.name+' — '+('论文全文' if filename.endswith('pdf') else 'HTML全文'))
   sub(a,'link:type',mime);sub(a,'z:path',resource=path.as_uri())
   attachments.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
  if not attachments:sub(paper,'dc:subject','缺少本地全文')
  authors=[]
  if (folder/'paper.pdf').exists():
   info=subprocess.run(['pdfinfo',str(folder/'paper.pdf')],capture_output=True,text=True).stdout
   a=re.search(r'^Author:\s*(.+)',info,re.M)
   if a and ';' in a[1]:
    authors=[x.strip() for x in a[1].split(';') if x.strip()]
    for author in authors:sub(paper,'dc:creator',author)
  note_id=ident+':notes';sub(paper,'dcterms:isReferencedBy',resource=note_id)
  note=node('bib:Memo',note_id)
  body=f'<p>来源：{html.escape(rel)}/README.md；导入日期：2026-11-04。</p>'
  body+=markdown.markdown(txt,extensions=['tables','fenced_code'])
  # Preserve usable external/local hyperlinks in the copied research notes.
  def fix(match):
   link=html.unescape(match[1])
   if not re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',link) and not link.startswith('#'):
    link=(folder/link).resolve().as_uri()
   return 'href="'+html.escape(link,quote=True)+'"'
  body=re.sub(r'href="([^"]+)"',fix,body)
  sub(note,'rdf:value',body)
  manifest.append({'folder':rel,'title':title,'url':url,'authors_from_pdf_metadata':authors,'attachments':attachments})
E.ElementTree(tree).write(out/'steering_references.rdf',encoding='utf-8',xml_declaration=True)
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('Prepared',len(manifest),'papers,',sum(len(r['attachments']) for r in manifest),'fulltexts; authors available for',sum(bool(r['authors_from_pdf_metadata']) for r in manifest))
