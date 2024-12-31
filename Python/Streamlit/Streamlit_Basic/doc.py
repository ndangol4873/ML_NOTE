import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

##### Text Utilities 

st.title('Startup Dashboard')

st.header('This is Header')

st.subheader('This subheader is below the header')

st.write('This is write')

st.markdown("""
### My Favorite Websites
- goggles
- tiktok
- amazon""")

st.code("""
def hello world():
    return 'Hello, World!'
""")

st.latex('x^2 + y^2 = 1')




###### Display Elements

df = pd.DataFrame(
    {
        'NAME': ['naresh','simrik', 'juju','sarvin', 'gaurav'],
        'DIGIT': [10, 20, 30, 40, 50],
        'CHAR': ['a', 'b', 'c', 'd', 'e']
    }
)

st.dataframe(df)


st.metric('Revenue', '3 L','3%')


st.json({
        'NAME': ['naresh','simrik', 'juju','sarvin', 'gaurav'],
        'DIGIT': [10, 20, 30, 40, 50],
        'CHAR': ['a', 'b', 'c', 'd', 'e']
    })


### Display Media

st.image('fb.png')
st.video('baba_love.mp4')


#### Creating Layouts
st.sidebar.header('Menu')
st.sidebar.selectbox('Select an option', ['Option 1', 'Option 2', 'Option 3'])
st.sidebar.multiselect('Select multiple options', ['Option 1', 'Option 2', 'Option 3'])

## Column 
st.columns(3)
col1, col2, col3 = st.columns(3)
with col1:st.image('fb.png')
with col2:st.image('fb.png')
with col3:st.image('fb.png')


## Showing Status
st.error('This is an error message')
st.warning('This is a warning message')
st.success('This is a success message')
st.info('This is an info message')

bar = st.progress(0)

for i in range(100):
    time.sleep(0.01)
    bar.progress(i)


## User Input   
name = st.text_input('Enter your name')
age = st.number_input('Enter your age')
dob = st.date_input('Enter your DOB')

st.button('Submit')