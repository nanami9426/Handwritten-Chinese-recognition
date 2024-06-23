<template>
  <div class="root">
  <div class="message" style="font-size: 14px; font-family: simsun;">
    {{ msg }}
    <span style="color: blue;text-decoration: underline;cursor: pointer;" @click="gopic(picurl)">{{ picurl }}</span>
  </div>
  <div>
    <input type="file" @change="handleFileUpload">
    <button @click="uploadImage" :disabled="pic_upload_btn">上传图片</button>
  </div>
  <div class="picctn" v-if="pic_is_upload">
    <div class="original_img" style="width: 500px;margin-top: 17px;">
      <img :src="picurl" style="width: 100%;">
    </div>
    <button @click="rec" style="margin-top: 17px;">识别图片</button>
    <div class="resctn" style="font-size: 14px; font-family: simsun;margin-top: 17px;">
      {{ restxt }}
      <div class="res_img" style="width: 500px;margin-top: 17px;" v-if="showres">
        <img :src="respicurl" style="width: 100%;">
      </div>
    </div>
  </div>
</div>
</template>
<script setup>
import { ref } from 'vue'
import axios from 'axios'
const msg = ref("")
const picurl = ref("")
const pic_upload_btn = ref(true)
const selectedFile = ref(null)
const pic_is_upload = ref(false)
const showres = ref(false)
const respicurl = ref("")
const restxt = ref("")
const gopic=(url)=>{
  window.open(url)
}


const handleFileUpload = (event) => {
  selectedFile.value = event.target.files[0]
  console.log(selectedFile.value);
  pic_upload_btn.value=false
}

const uploadImage = async () => {
  try {
    const formData = new FormData()
    formData.append('image', selectedFile.value)

    const response = await axios.post('http://127.0.0.1:5000/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })

    console.log('上传成功', response.data)
    msg.value = '上传成功，图片地址为'
    picurl.value = response.data
    pic_is_upload.value = true
  } catch (error) {
    console.error('上传失败', error)
    msg.value = `上传失败，${error}`
    picurl.value = ""

  }
}
const rec = async()=>{
  const data = {
    picsrc:picurl.value
  }
  const res = await axios.post('http://127.0.0.1:5000/api/rec',data)
  console.log(res);
  msg.value = '识别成功，识别后图片地址为'
  picurl.value = res.data.respic
  restxt.value = res.data.warps
}
</script>


<style scoped>
.root{
  margin-left: 27px;
  margin-top: 27px;
}
.message{
  width: 100%;
  height: 33px;
}
</style>