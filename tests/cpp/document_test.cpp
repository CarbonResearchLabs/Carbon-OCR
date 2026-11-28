// The document stream (carbon/document.h): a page reads the same through it as through its steps one by one,
// whatever order the pages finish in. On tests/data/fake-set; PDFs through the helper processes.

#include "carbon/document.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "carbon/error.h"
#include "carbon/image.h"
#include "carbon/textlayer.h"

namespace carbon {
namespace {

const std::filesystem::path kData = CARBON_TEST_DATA;
const std::filesystem::path kSet = kData / "fake-set";

// A page's JSON without its timings.
std::string without_ms(const std::string& json) { return json.substr(0, json.rfind(", \"ms\": {")); }

EngineOptions two_workers() {
  EngineOptions o;
  o.pool.workers = 2;
  return o;
}

// Images, given and as files and bytes, each its tag back with the engine's own reading of it; a file that is not
// there is its page's error alone.
TEST(DocumentStream, ReadsImagesAsTheEngineReadsThem) {
  DocumentOptions d;
  d.engine = two_workers();
  const GrayImage page = load_image(kSet / "page.png");
  const std::string want = without_ms(Engine(kSet, d.engine).ocr(page.view()).json());
  DocumentStream stream(kSet, d);
  EXPECT_FALSE(stream.cuda());
  EXPECT_EQ(stream.workers(), 2);
  stream.image(GrayImage(page), 10);
  stream.image_file(kSet / "page.png", 11);
  stream.image_file(kSet / "no-such-page.png", 12);
  std::ifstream f(kSet / "page.png", std::ios::binary);
  auto bytes = std::make_shared<const std::string>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  stream.encoded(bytes, 13);
  stream.encoded(std::make_shared<const std::string>("not an image"), 14);
  stream.close();
  EXPECT_THROW(stream.image(GrayImage(page), 15), Error);
  std::map<uint64_t, DocumentStream::Done> got;
  while (auto done = stream.next()) got.emplace(done->tag, std::move(*done));
  ASSERT_EQ(got.size(), 5u);
  for (uint64_t t : {10, 11, 13}) {
    ASSERT_FALSE(got[t].error) << t;
    EXPECT_EQ(without_ms(got[t].page.json()), want) << t;
    EXPECT_FALSE(got[t].from_text_layer);
  }
  EXPECT_TRUE(got[12].error);
  EXPECT_TRUE(got[12].unreadable);
  EXPECT_TRUE(got[14].error);
  EXPECT_TRUE(got[14].unreadable);
}

#if defined(CARBON_PDF_HELPER)
// A page read from its text layer whose picture holds a line (figure.pdf: a bar the fake set reads as "aaa"),
// through the stream, against its steps one after another as carbon-ocr ocr took them after its stream: the layer's
// page, its words in the detector's lines, its pictures' page read as given by an engine that never turns or levels
// a page, those lines added, and the order. The text pages of text.pdf alongside, and an image.
TEST(DocumentStream, ReadsATextLayersPageAsItsStepsOneByOne) {
  PdfPool pool(1, CARBON_PDF_HELPER);
  const EngineOptions o = two_workers();
  for (const PictureRead mode : {PictureRead::kDetect, PictureRead::kFound, PictureRead::kOff}) {
    DocumentOptions d;
    d.engine = o;
    d.pictures = mode;
    DocumentStream stream(kSet, d);
    const PdfPage pg = pool.page(kData / "figure.pdf", 0, 200 / 72.0, stream.pdf_want());
    // the steps one by one
    const TextLayer layer = read_text_layer(pg.chars, pg.image.view());
    ASSERT_TRUE(layer.check.usable);
    OcrPage want = page_from_tagged_layer(layer, pg.structure, pg.image.width, pg.image.height);
    ASSERT_TRUE(reads_across(layer));
    LineFinder finder(kSet, o);
    const OcrPage found = finder.detect(pg.image.view());
    std::vector<LineBox> boxes;
    for (const OcrLine& l : found.lines) boxes.push_back(l.box);
    TextLayer detected = layer;
    lines_from_detection(detected, boxes);
    want = page_from_text_layer(detected, pg.image.width, pg.image.height);
    if (mode != PictureRead::kOff) {
      want.pictures = pg.pictures;
      std::vector<LineBox> to_read = pictures_to_read(want.pictures, pg.image.width, pg.image.height);
      for (const LineBox& b : uncovered_ink(pg.image.view(), layer, pg.chars, want.pictures)) to_read.push_back(b);
      ASSERT_FALSE(to_read.empty());
      const GrayImage pictures = pictures_only(pg.image.view(), to_read);
      EngineOptions given = o;
      given.turn_probability = 2.0f;
      given.deskew = false;
      Engine engine(kSet, given);
      OcrPage read;
      if (mode == PictureRead::kDetect) {
        read = engine.ocr(pictures.view());
      } else {  // the lines found on the whole page, read on the pictures' page
        PageTask task;
        task.found = std::make_shared<const OcrPage>(picture_lines(found, want, to_read));
        ASSERT_FALSE(task.found->lines.empty());
        engine.submit(GrayImage(pictures), task);
        read = *engine.next();
      }
      add_picture_lines(want, read);
    }
    std::vector<OcrPage> ordered = {want};
    Engine(kSet, o).order(ordered);
    // through the stream, with other pages beside it
    stream.pdf_page(pool.page(kData / "text.pdf", 0, 200 / 72.0, stream.pdf_want()), 1);
    stream.pdf_page(PdfPage(pg), 0);
    stream.image(load_image(kSet / "page.png"), 2);
    stream.pdf_page(pool.page(kData / "text.pdf", 1, 200 / 72.0, stream.pdf_want()), 3);
    stream.close();
    std::map<uint64_t, DocumentStream::Done> got;
    while (auto done = stream.next()) got.emplace(done->tag, std::move(*done));
    ASSERT_EQ(got.size(), 4u);
    ASSERT_FALSE(got[0].error);
    EXPECT_TRUE(got[0].from_text_layer);
    EXPECT_EQ(got[0].pictures_read, mode != PictureRead::kOff);
    EXPECT_EQ(got[0].page.source, mode == PictureRead::kOff ? "text layer" : "hybrid");
    EXPECT_EQ(without_ms(got[0].page.json()), without_ms(ordered[0].json())) << static_cast<int>(mode);
    EXPECT_EQ(got[1].page.source, "text layer");
    EXPECT_EQ(got[3].page.source, "text layer");
    EXPECT_EQ(got[2].page.source, "ocr");
  }
}

// Turned layers and bad-code runs through the stream: turned.pdf's pages, whose layers run up the page, down it and
// upside down, read upright, each its rotation the turn (its boxes the upright page's); badruns.pdf's line of bad codes set aside and
// read by OCR on the pictures' page, the same with the lines found as with that page detected, its sound lines
// kept.
TEST(DocumentStream, ReadsTurnedLayersUprightAndSetAsideLinesByOcr) {
  PdfPool pool(1, CARBON_PDF_HELPER);
  std::map<int, std::string> by_mode;
  for (const PictureRead mode : {PictureRead::kFound, PictureRead::kDetect}) {
    DocumentOptions d;
    d.engine = two_workers();
    d.pictures = mode;
    DocumentStream stream(kSet, d);
    EXPECT_TRUE(stream.pdf_want() & kPdfUpright);
    const int turned = pool.pages(kData / "turned.pdf");
    ASSERT_GE(turned, 4);
    for (int p = 0; p < turned; ++p) stream.pdf_page(pool.page(kData / "turned.pdf", p, 200 / 72.0, stream.pdf_want()), p);
    stream.pdf_page(pool.page(kData / "badruns.pdf", 0, 200 / 72.0, stream.pdf_want()), 100);
    stream.close();
    std::map<uint64_t, DocumentStream::Done> got;
    while (auto done = stream.next()) got.emplace(done->tag, std::move(*done));
    const int rotations[] = {270, 0, 180, 180};
    for (int p = 0; p < 4; ++p) {
      ASSERT_FALSE(got[p].error) << p;
      EXPECT_TRUE(got[p].from_text_layer) << p;
      EXPECT_EQ(got[p].page.rotation, rotations[p]) << "turned.pdf page " << p;
    }
    const DocumentStream::Done& bad = got[100];
    ASSERT_FALSE(bad.error);
    EXPECT_TRUE(bad.from_text_layer);
    EXPECT_TRUE(bad.pictures_read) << "the set-aside line read by OCR";
    EXPECT_NE(bad.page.source, "ocr");
    by_mode[static_cast<int>(mode)] = without_ms(bad.page.json());
  }
  EXPECT_EQ(by_mode[static_cast<int>(PictureRead::kFound)], by_mode[static_cast<int>(PictureRead::kDetect)]);
}

// A PDF page whose layer fails its checks (the text layer off here) goes to OCR, its reason kept.
TEST(DocumentStream, APageWithoutItsLayerIsOcrd) {
  PdfPool pool(1, CARBON_PDF_HELPER);
  DocumentOptions d;
  d.engine = two_workers();
  d.text_layer = false;
  DocumentStream stream(kSet, d);
  EXPECT_EQ(stream.pdf_want(), static_cast<unsigned>(kPdfImage));
  PdfPage pg = pool.page(kData / "text.pdf", 0, 200 / 72.0, stream.pdf_want());
  const std::string want = without_ms(Engine(kSet, d.engine).ocr(pg.image.view()).json());
  stream.pdf_page(std::move(pg), 5);
  stream.close();
  const auto done = stream.next();
  ASSERT_TRUE(done.has_value());
  EXPECT_EQ(done->tag, 5u);
  EXPECT_FALSE(done->from_text_layer);
  EXPECT_EQ(without_ms(done->page.json()), want);
  EXPECT_FALSE(stream.next().has_value());
}
#endif

}  // namespace
}  // namespace carbon
