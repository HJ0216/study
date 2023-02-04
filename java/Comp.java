public class Comp {
	public static void main(String[] args) {
		char ch = 'e'; // 66

//		System.out.println((int)'A'); // 65
//		System.out.println((int)'a'); // 97
		
//		int result = ch >= 97 ? ch-32 : ch+32;
//		int result = (ch >= 'A' && ch <='Z') ? ch+32 : ch-32; // char result: 2byte, int ch+2: 4byte

		int sub = 'a' - 'A';
		int result = (ch >= 'A' && ch <='Z') ? ch+sub : ch-sub;
		
		System.out.println(ch + "->" + (char)result);
		
	}
	
}