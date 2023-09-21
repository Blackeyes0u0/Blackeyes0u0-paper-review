import markdown

# mkfile = ''
mkfile = './Vision/ViT/vit.md'
url = 'https://github.com/Blackeyes0u0/Blackeyes0u0-paper-review/blob/master/papers/Vision/ViT/'

# 마크다운 파일 읽기
with open(mkfile, "r") as f:
    markdown_content = f.read()

# 마크다운 파일 수정
# a = markdown_content.find('png')
# pnglist = re.findall('png',markdown_content)

markdown_content = markdown_content.replace('Alt text](','Alt text](' + url )
markdown_content = markdown_content.replace('png', 'png' + "?raw=true")

# 수정된 마크다운 파일 저장
with open("velog.md", "w") as f:
    f.write(markdown_content)
