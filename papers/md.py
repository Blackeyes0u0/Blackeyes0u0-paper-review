import markdown

def md_to_html(md_file):
    with open(md_file, 'r') as f:
        md_content = f.read()

    html_content = markdown.markdown(md_content)

    with open('output.html', 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    md_file = 'input.md'
    md_to_html(md_file)
