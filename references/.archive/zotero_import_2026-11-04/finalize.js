const base = '/home/nubot/humanlike_drive/references/.archive/zotero_import_2026-11-04/';
const mapping = JSON.parse(await Zotero.File.getContentsAsync(base + 'item_mapping.json'));
const result = { categorized: [], repairedAttachments: [] };
for (const entry of mapping) {
    const item = await Zotero.Items.getByLibraryAndKeyAsync(1, entry.key);
    if (!item || item.getField('extra').indexOf('Local reference: ' + entry.folder) === -1) {
        throw new Error('Imported item identity mismatch: ' + entry.key);
    }
    item.addToCollection(entry.collectionID);
    await item.saveTx();
    result.categorized.push({key: item.key, collectionID: entry.collectionID});
}
const patternDexAttachment = await Zotero.Items.getByLibraryAndKeyAsync(1, 'DQA9E6PF');
if (!(await patternDexAttachment.fileExists())) {
    const dir = Zotero.Attachments.getStorageDirectory(patternDexAttachment);
    const paths = await IOUtils.getChildren(dir.path);
    const pdfs = paths.filter(p => p.toLowerCase().endsWith('.pdf'));
    if (pdfs.length !== 1) throw new Error('Ambiguous PatternDex attachment path');
    await patternDexAttachment.relinkAttachmentFile(pdfs[0]);
    result.repairedAttachments.push({key:patternDexAttachment.key,path:pdfs[0]});
}
await Zotero.File.putContentsAsync(base + 'finalize_result.json', JSON.stringify(result, null, 2));
await Zotero.getMainWindow().ZoteroPane.collectionsView.selectByID('C36');
return result;
