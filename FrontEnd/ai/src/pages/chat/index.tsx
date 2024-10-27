import React, { useEffect, useRef, useState } from 'react';
import styles from './styles.module.scss';
import { useSearchParams } from 'react-router-dom';
import { SearchOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { Input, App, Avatar, Spin, Modal } from 'antd';
import { useAiStore } from '@/store';
import request from '@/utils/request.ts';
import { REQUEST_URLS } from '@/config/requet-urls.ts';

const ChatPage: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const { message } = App.useApp();
  const [searchParams] = useSearchParams();
  const query = searchParams.get('q');
  const ref = useRef<HTMLDivElement | null>(null);

  const [loading, setLoading] = useState(false);
  const [input, setInput] = useState('');

  const [videoPath, setVideoPath] = useState('');
  const phone = useAiStore(s => s.phone);
  const chatList = useAiStore(state => state.chatList);
  const setChatList = useAiStore(state => state.setChatList);

  const onScroll = () => {
    setTimeout(() => {
      if (ref.current) {
        ref.current.scrollTo({
          top: 50000,
          behavior: 'smooth',
        });
      }
    }, 200);
  };

  const onConfirm = () => {
    if (!input) {
      message.warning('输入东西啊你倒是');
      return;
    }
    if (loading) {
      message.warning('正在生成中');
      return;
    }
    setInput('');
    setChatList({
      id: Date.now() + 'question',
      content: input,
      type: 'question',
    });

    setLoading(true);
    onScroll();
    request
      .post(REQUEST_URLS.chat, {
        phone_number: phone,
        message: input,
      })
      .then(v => {
        setChatList({
          id: Date.now() + 'answer',
          content: v?.data?.message || '生成失败，请稍后再试',
          type: 'answer',
          videoPath: v?.data?.video_path,
          linkPath: v?.data?.link_path,
        });
        onScroll();
      })
      .catch(e => {
        message.error(typeof e === 'string' ? e || '网络错误' : e.toString());
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    if (query) {
      setInput(query);
    }
  }, [query]);

  useEffect(() => {
    setTimeout(() => {
      if (videoRef.current) {
        if (!videoPath) {
          videoRef.current.pause();
          videoRef.current.currentTime = 0;
        } else {
          videoRef.current.play();
        }
      }
    }, 500);
  }, [videoPath]);

  return (
    <div className={styles.container}>
      <div className={styles.chat} ref={ref}>
        {chatList.map(item => (
          <div key={item.id} className={item.type === 'question' ? styles.chatRContainer : styles.chatLContainer}>
            <div className={styles.chatContent}>
              {item.type === 'answer' ? <Avatar style={{ backgroundColor: '#87d068', flex: '0 0 auto' }}>A</Avatar> : null}
              <div className={styles.content} dangerouslySetInnerHTML={{ __html: item.content }} />
              {item.type === 'answer' && item.videoPath ? (
                <PlayCircleOutlined style={{ fontSize: 18, cursor: 'pointer' }} onClick={() => setVideoPath(item.linkPath || '')} />
              ) : null}
              {item.type === 'question' ? (
                <Avatar style={{ backgroundColor: '#fde3cf', color: '#f56a00', flex: '0 0 auto' }}>Q</Avatar>
              ) : null}
            </div>
          </div>
        ))}
        {loading ? (
          <div className={styles.chatContent}>
            <Avatar style={{ backgroundColor: '#87d068', flex: '0 0 auto' }}>A</Avatar>
            <div className={styles.content}>
              <Spin />
              <span style={{ marginTop: -2, marginLeft: 8 }}>Let me think...</span>
            </div>
          </div>
        ) : null}
      </div>
      <Input
        value={input}
        className={styles.input}
        size="large"
        addonAfter={
          <div className={styles.search} onClick={() => onConfirm()}>
            <SearchOutlined />
          </div>
        }
        maxLength={500}
        placeholder={import.meta.env.VITE_APP_SLOGAN}
        allowClear
        onChange={e => setInput(e.target.value)}
        onPressEnter={onConfirm}
      />
      <Modal title="Video Player" open={!!videoPath} onOk={() => setVideoPath('')} onCancel={() => setVideoPath('')} footer={null}>
        <video key={videoPath} ref={videoRef} width="100%" height={360} controls>
          <source src={videoPath} type="video/mp4" />
        </video>
      </Modal>
    </div>
  );
};

export default ChatPage;
